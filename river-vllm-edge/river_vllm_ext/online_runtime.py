"""In-process experimental routing and phase-separated scheduling for vLLM V1.

Requires synchronous scheduling, PP/TP=1, no prefix caching, no speculation,
and eager execution. No vLLM source files are patched.
"""

import time


class RoutePolicy:
    def __init__(self, mode="batch", low=8, high=16):
        if mode not in {"fp", "ptq", "mixed", "batch", "conservative"} or not 0 < low < high:
            raise ValueError("Invalid route policy or crossover thresholds")
        self.mode, self.low, self.high = mode, low, high
        self.use_ptq = True
        self.mixed_eligibility = False

    def update(self, active_decode, eligible_decode=None):
        self.mixed_eligibility = eligible_decode is not None and 0 < eligible_decode < active_decode
        if active_decode >= self.high:
            self.use_ptq = False
        elif active_decode <= self.low:
            self.use_ptq = True

    def select(self, extra_args):
        # Eligibility is externally supplied, not an implemented River gate.
        extra_args = extra_args or {}
        if self.mode == "fp":
            return "fp"
        if self.mode == "ptq":
            return "ptq"
        if not extra_args.get("riveredge_allow_ptq", False):
            return "fp"
        # Initial measured mixed-route overhead exceeds the homogeneous FP
        # step cost. This optional fallback avoids paying for both tails.
        if self.mode == "conservative" and self.mixed_eligibility:
            return "fp"
        return "ptq" if self.mode == "mixed" or self.use_ptq else "fp"


class OnlineRuntime:
    def __init__(self, llm, policy, grouped=False, fairness_steps=4,
                 max_wait_s=0.2):
        core_client = llm.llm_engine.engine_core
        self.core = getattr(core_client, "engine_core", core_client)
        self.runner = self.core.model_executor.driver_worker.model_runner
        self.model = self.runner.model
        self.model.logit_routes = None
        self.scheduler = self.core.scheduler
        if not hasattr(self.model, "row_routes"):
            raise ValueError("Expected eager RiverEdgeOnlineForCausalLM")
        config = self.runner.vllm_config
        if (config.scheduler_config.async_scheduling
                or config.speculative_config is not None
                or config.cache_config.enable_prefix_caching
                or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("Unsupported scheduler/cache/parallel configuration")
        if fairness_steps < 1 or max_wait_s <= 0:
            raise ValueError("Fairness bounds must be positive")
        self.policy, self.grouped = policy, grouped
        self.fairness_steps, self.max_wait_s = fairness_steps, max_wait_s
        self.steps = []
        self.pending = None
        self.last_service = {}
        self.last_group_step = {"prefill": -1, "fp": -1, "ptq": -1}
        self.original_schedule = self.scheduler.schedule
        self.original_execute = self.runner.execute_model
        self.scheduler.schedule = self.schedule
        self.runner.execute_model = self.execute
        self.hook = self.model.register_forward_pre_hook(self.before_forward,
                                                        with_kwargs=True)

    def phase(self, request):
        if request.num_computed_tokens < request.num_prompt_tokens:
            return "prefill"
        return self.policy.select(request.sampling_params.extra_args)

    def schedule(self, *args, **kwargs):
        scheduler = self.scheduler
        active = sum(r.num_computed_tokens >= r.num_prompt_tokens
                     for r in scheduler.running)
        eligible = sum(r.num_computed_tokens >= r.num_prompt_tokens
                       and (r.sampling_params.extra_args or {}).get("riveredge_allow_ptq", False)
                       for r in scheduler.running)
        self.policy.update(active, eligible)
        now = time.monotonic()
        live_ids = set(scheduler.requests)
        self.last_service = {key: value for key, value in self.last_service.items()
                             if key in live_ids}
        for request in scheduler.requests.values():
            self.last_service.setdefault(request.request_id, now)
        selected = "native"
        if not self.grouped:
            output = self.original_schedule(*args, **kwargs)
        else:
            from vllm.v1.core.sched.request_queue import create_request_queue

            groups = {name: [] for name in ("prefill", "fp", "ptq")}
            for request in scheduler.running:
                groups[self.phase(request)].append(request)
            has_capacity = len(scheduler.running) < scheduler.max_num_running_reqs
            if has_capacity:
                groups["prefill"].extend(list(scheduler.waiting))
                groups["prefill"].extend(list(scheduler.skipped_waiting))
            available = [name for name, requests in groups.items() if requests]
            if not available:
                output = self.original_schedule(*args, **kwargs)
            else:
                def score(name):
                    age = max(now - self.last_service[r.request_id]
                              for r in groups[name])
                    gap = len(self.steps) - self.last_group_step[name]
                    budgets = [(r.sampling_params.extra_args or {}).get(
                        "riveredge_ttft_slo" if name == "prefill" else "riveredge_tpot_slo",
                        self.max_wait_s) for r in groups[name]]
                    overdue = age >= min(budgets) or gap >= self.fairness_steps
                    return (overdue, age if overdue else 0,
                            {"ptq": 2, "prefill": 1, "fp": 0}[name])

                selected = max(available, key=score)
                self.last_group_step[selected] = len(self.steps)
                held = [r for r in scheduler.running if self.phase(r) != selected]
                scheduler.running = [r for r in scheduler.running
                                     if self.phase(r) == selected]
                waiting, skipped = scheduler.waiting, scheduler.skipped_waiting
                limit = scheduler.max_num_running_reqs
                scheduler.max_num_running_reqs = limit - len(held)
                if selected != "prefill":
                    scheduler.waiting = create_request_queue(scheduler.policy)
                    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
                try:
                    output = self.original_schedule(*args, **kwargs)
                finally:
                    scheduler.running.extend(held)
                    scheduler.max_num_running_reqs = limit
                    if selected != "prefill":
                        # Preserve any requests preempted by the native allocator.
                        for request in reversed(list(scheduler.waiting)):
                            waiting.prepend_request(request)
                        for request in reversed(list(scheduler.skipped_waiting)):
                            skipped.prepend_request(request)
                        scheduler.waiting, scheduler.skipped_waiting = waiting, skipped
        if output.preempted_req_ids:
            raise RuntimeError(
                "Online reference does not replay historical routes after KV "
                "preemption; increase --kv-cache-bytes or lower --batch-size")
        waits = {rid: now - self.last_service[rid]
                 for rid in output.num_scheduled_tokens}
        for rid in output.num_scheduled_tokens:
            self.last_service[rid] = now
        self.steps.append({"step": len(self.steps), "time_s": now,
                           "selected_group": selected, "active_decode": active,
                           "scheduled_tokens": dict(output.num_scheduled_tokens),
                           "service_gap_s": waits, "routes": {},
                           "graph": "eager"})
        return output

    def execute(self, scheduler_output, *args, **kwargs):
        self.pending = scheduler_output
        try:
            return self.original_execute(scheduler_output, *args, **kwargs)
        finally:
            self.pending = None
            self.model.row_routes = None
            self.model.logit_routes = None

    def before_forward(self, module, args, kwargs):
        if self.pending is None:
            return
        batch = self.runner.input_batch
        routes, logit_routes, request_routes = [], [], {}
        for index, rid in enumerate(batch.req_ids):
            count = self.pending.num_scheduled_tokens[rid]
            computed = int(batch.num_computed_tokens_cpu[index])
            prompt_length = int(batch.num_prompt_tokens[index])
            state = self.runner.requests[rid]
            route = self.policy.select(state.sampling_params.extra_args)
            # Explicit token position protects one-token prompts and chunked
            # prefill; max_query_len/max_seq_len heuristics are insufficient.
            values = [computed + offset >= prompt_length and route == "ptq"
                      for offset in range(count)]
            routes.extend(values)
            logit_routes.append(values[-1])
            request_routes[rid] = {"route": route if any(values) else "fp",
                                   "phase": "decode" if computed >= prompt_length else "prefill",
                                   "tokens": count}
        positions = kwargs.get("positions")
        if positions is None:
            positions = args[1]
        if len(routes) != positions.shape[0]:
            raise RuntimeError("Unsupported padding or runner token arrangement")
        module.row_routes = routes
        module.logit_routes = logit_routes
        self.steps[-1]["routes"] = request_routes

    def close(self):
        self.hook.remove()
        self.scheduler.schedule = self.original_schedule
        self.runner.execute_model = self.original_execute
        self.model.row_routes = None
        self.model.logit_routes = None
