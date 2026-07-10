# RiverEdge To-do List 2026-07-09

## 0. 当前环境约定

- 仓库根目录：`/home/orin/zjc/vllm`
- RiverEdge 目录：`/home/orin/zjc/vllm/RiverEdge`
- 模型目录：`/media/orin/Data/models`，容器内为 `/models`
- HF cache：`/media/orin/Data/huggingface`，River 容器内为 `/disk/dataset/huggingface`
- River 容器：`river-llama32-8b`、`river-bench-t`
- River 权重：`/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16`
- vLLM baseline image：`vllm/vllm-openai:latest-aarch64`

所有新增项目代码建议放在：

```text
/home/orin/zjc/vllm/RiverEdge/river-vllm-edge/
```

不要直接修改 `rivier/` 的核心模型实现或 vLLM internals，除非前置 gate 已通过。

## 1. P0：先明确研究对象

目标是判断 RiverEdge 应该走哪条路线：

- 路线 A：真正 early exit，固定 checkpoint 后跳过后续层。
- 路线 B：adaptive-precision routed continuation，固定 checkpoint 后继续执行 PTQ tail 或 FP tail。

当前计划更接近路线 B。路线 B 的收益来自 PTQ tail 比 FP tail 更快，而不是来自跳过层。

交付物：

- `experiments/00_problem_definition/problem_statement.md`
- `experiments/00_problem_definition/go_no_go_gates.md`

## 2. P1：环境与基线记录

在宿主机记录：

```bash
uname -a
cat /etc/os-release
df -h
free -h
docker ps
docker images
```

在 River 容器内记录：

```bash
sudo docker exec -it river-llama32-8b bash
python3 --version
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python3 -c "import transformers; print(transformers.__version__)"
cd /workspace/vllm/rivier/lm-evaluation
python -m pip list | grep -E "torch|transformers|hqq|lm-eval|numpy"
```

交付物：

- `experiments/01_env/env_report.md`
- `experiments/01_env/docker_report.md`

## 3. P2：River 原始行为验证

先不接 vLLM，继续使用 River/lm-evaluation 环境。

必须完成：

- 完整 `mmlu_abstract_algebra` early-exit vs full model baseline。
- 至少 2-3 个 MMLU 子任务。
- 一个小样本 GSM8K，仅作为生成任务 sanity check。
- 保存 exit layer histogram、avg exit layer、full model ratio、accuracy。

建议命令模板：

```bash
sudo docker exec -it river-llama32-8b bash
cd /workspace/vllm/rivier/lm-evaluation
CUDA_VISIBLE_DEVICES=0 QUANT_BACKEND=n bash ./eval_8B.sh mmlu_abstract_algebra 0.5
CUDA_VISIBLE_DEVICES=0 QUANT_BACKEND=n bash ./eval_8B.sh mmlu_abstract_algebra 1.01
```

交付物：

- `experiments/02_river_original/accuracy_summary.csv`
- `experiments/02_river_original/exit_distribution.csv`
- `experiments/02_river_original/notes.md`

## 4. P3：确认权重结构是否支持双 tail

这是最关键的可行性检查。需要确认：

- 当前 checkpoint 是否有完整 FP backbone。
- 是否有可独立执行的 PTQ tail。
- `exit_modules` 是否能被解释为 `layers k+1..L` 的 PTQ tail。
- FP tail 与 PTQ tail 是否输出同 shape hidden states。
- 两条 tail 是否都能写入兼容 KV。

交付物：

- `experiments/03_weight_structure/model_structure_report.md`
- `experiments/03_weight_structure/module_tree.txt`
- `experiments/03_weight_structure/state_dict_keys.txt`

Go/No-Go：

- 如果不能构造 `shared + FP tail` 和 `shared + PTQ tail`，暂停 vLLM 集成。

## 5. P4：PyTorch split model reference

先在 PyTorch/River 内实现 reference，不进入 vLLM。

模式：

- `full_fp`
- `shared_fp_plus_fp_tail`
- `shared_fp_plus_ptq_tail`
- `all_ptq_tail`

验证：

- 单步 forward shape 一致。
- 多步 decode 不报错。
- `shared_fp_plus_fp_tail` 与 full model 输出接近。
- batch size 1/2/4 下测 tail latency。

交付物：

- `river-vllm-edge/river_vllm_ext/models/split_model_reference.py`
- `river-vllm-edge/scripts/test_split_forward.py`
- `experiments/04_split_reference/tail_latency.csv`
- `experiments/04_split_reference/quality_sanity.csv`

Go/No-Go：

- PTQ tail 在 AGX 上至少应比 FP tail 快 1.3x，否则不进入 scheduler 方向。

## 6. P5：vLLM baseline 与 source build

只有 P3/P4 通过后再做。

先跑官方 image baseline：

```bash
sudo docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /media/orin/Data/models:/models \
  -v /home/orin/zjc/vllm:/workspace/vllm \
  vllm/vllm-openai:latest-aarch64 \
    --model /models/Qwen3.5-0.8B \
    --served-model-name qwen3.5-0.8b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 4
```

然后再建立 source-based vLLM 环境：

```text
RiverEdge/river-vllm-edge/third_party/vllm/
```

交付物：

- `experiments/05_vllm_baseline/official_image_summary.csv`
- `experiments/06_source_vllm/source_selection.md`
- `experiments/06_source_vllm/build_log.txt`

## 7. P6：vLLM custom model，先不改 scheduler

目标是让 vLLM 能跑 custom model 的三种静态模式：

- `full_fp`
- `static_fp_tail`
- `static_ptq_tail`

暂不做 route-aware queue。

交付物：

- `river-vllm-edge/river_vllm_ext/plugin.py`
- `river-vllm-edge/river_vllm_ext/models/routed_llama.py`
- `experiments/07_vllm_custom_model/summary.csv`

Go/No-Go：

- custom full-FP 相比 source-built native full-FP 性能损失小于 10%。

## 8. P7：naive routed execution

实现同步版本：

```text
shared layers 1-k
  -> route decision
  -> PTQ tail batch + FP tail batch
  -> merge
  -> sampler
```

route policy：

- `all_ptq`
- `all_fp`
- `random`
- `river_gate`

交付物：

- `experiments/08_naive_routed/summary.csv`
- `experiments/08_naive_routed/route_ratio.csv`

Go/No-Go：

- 如果 naive routed 在 PTQ-heavy workload 下仍无收益，暂停 microbatch runtime。

## 9. P8：route-aware microbatch runtime

这是高风险阶段。只有 P7 有明确收益时进入。

实现：

- `Q_pre_decode`
- `Q_ptq_decode`
- `Q_fp_decode`
- PTQ 完成后可先 sampling 并回到 pre-decode。
- FP queue 有 fairness，避免 starvation。

交付物：

- `experiments/09_microbatch_runtime/summary.csv`
- `experiments/09_microbatch_runtime/queue_trace.jsonl`
- `experiments/09_microbatch_runtime/tpot_percentiles.csv`

## 10. P9：scheduler / CUDA Graph instrumentation

先做 instrumentation，再做优化。

记录：

- phase
- batch size
- graph used/eager used
- latency
- queue wait time
- route ratio

交付物：

- `experiments/10_scheduler_graph/graph_trace.jsonl`
- `experiments/10_scheduler_graph/summary.csv`

## 11. 投稿前结果组织

至少需要以下对照：

- native vLLM FP baseline
- custom full-FP overhead
- static PTQ tail
- naive routed
- route-aware microbatch
- scheduler fairness ablation

论文主指标：

- TPS
- mean TPOT
- P50/P95/P99 TPOT
- quality-corrected speedup
- graph hit rate
- queue wait time
- energy/token，如可测

