# RiverEdge Project Checklist 2026-07-09

用于监督项目进度。每个阶段只有在通过 Go/No-Go 条件后，才建议进入下一阶段。

## A. 项目定义

- [ ] 明确 RiverEdge 当前采用的路线：true early exit 或 routed continuation。
- [ ] 写清楚本项目的系统问题：batch divergence、paged KV、CUDA Graph、TPOT。
- [ ] 定义主要指标：TPS、TPOT、P95/P99、quality-corrected speedup。
- [ ] 定义不可接受的质量下降阈值。
- [ ] 完成 `experiments/00_problem_definition/problem_statement.md`。
- [ ] 完成 `experiments/00_problem_definition/go_no_go_gates.md`。

通过标准：

- 项目贡献不再混淆 “跳过层 early exit” 与 “PTQ/FP tail routed continuation”。

## B. 环境盘点

- [ ] 记录宿主机 OS、磁盘、内存、Docker 状态。
- [ ] 记录 `river-llama32-8b` 容器 Python/PyTorch/CUDA/Transformers 版本。
- [ ] 记录 `river-bench-t` 容器用途和环境差异。
- [ ] 记录 vLLM baseline image 信息：`vllm/vllm-openai:latest-aarch64`。
- [ ] 确认模型路径 `/media/orin/Data/models` 与容器内 `/models` 一致。
- [ ] 确认 HF cache 路径 `/media/orin/Data/huggingface` 可用。
- [ ] 完成 `experiments/01_env/env_report.md`。

通过标准：

- 任意实验结果都能追溯到 Docker image、模型路径、脚本路径和关键版本。

## C. River 原始行为

- [x] 完整 `mmlu_abstract_algebra` early-exit vs full-model baseline 已跑通。
- [x] 记录完整 `mmlu_abstract_algebra` accuracy。
- [x] 记录完整 `mmlu_abstract_algebra` exit distribution。
- [ ] 保存 per-sample 明细，用于判断 early-exit 与 baseline 是否逐样本一致。
- [ ] 补 2-3 个 MMLU 子任务。
- [ ] 补 HellaSwag 或 ARC 子任务。
- [ ] GSM8K 小样本 sanity check。
- [ ] 汇总 `experiments/02_river_original/accuracy_summary.csv`。
- [ ] 汇总 `experiments/02_river_original/exit_distribution.csv`。

当前已知结果：

| Task | Setting | Samples | Threshold | Acc | Avg Exit Layer | Full Model |
|---|---|---:|---:|---:|---:|---:|
| mmlu_abstract_algebra | early exit | 100 | 0.5 | 0.3500 | 3.00 | 0.00% |
| mmlu_abstract_algebra | baseline | 100 | 1.01 | 0.3500 | 32.00 | 100.00% |

通过标准：

- 至少三个任务上确认 exit 分布与质量预算。

## D. 权重结构可行性

- [x] 导出 checkpoint module tree。
- [x] 导出 state_dict keys。
- [x] 确认是否存在完整 FP backbone。
- [x] 确认是否存在可独立执行的 PTQ tail。
- [x] 确认 `exit_modules` 是否能作为 `layers k+1..L` tail。
- [x] 确认 PTQ tail 和 FP tail hidden state shape 一致。
- [ ] 确认 PTQ tail 和 FP tail KV layout 一致或可转换。
- [x] 完成 `experiments/03_weight_structure/model_structure_report.md`。

Go/No-Go：

- 如果不能构造 `shared + FP tail` 和 `shared + PTQ tail`，暂停 vLLM 集成。

## E. PyTorch Split Reference

- [x] 实现 `full_fp` reference。
- [x] 实现 `shared_fp_plus_fp_tail`。
- [x] 实现 `shared_fp_plus_ptq_tail`。
- [x] 单 token forward shape 测试通过。
- [x] 多步 decode 测试通过。
- [ ] 比较 `shared_fp_plus_fp_tail` 与 full model 输出差异。
- [x] batch size 1/2/4 tail latency 测试完成。
- [x] 记录 FP tail latency。
- [x] 记录 PTQ tail latency。
- [x] 完成 `experiments/04_split_reference/tail_latency.csv`。
- [ ] 完成 `experiments/04_split_reference/quality_sanity.csv`。

Go/No-Go：

- PTQ tail 至少在目标 batch size 下稳定快于 FP tail。
- 建议最低门槛：`speedup >= 1.3x`。

当前补充结果：

- PyTorch routed reference 已实现：prefill full FP，decode shared FP prefix 后 route 到 PTQ/FP tail。
- 默认 `k=3`、`threshold=0.5`，prompt 64、生成 16 token 的 auto route 全部选择 PTQ。
- mixed-cache reference 中 force PTQ decode TPS 为 12.73，force FP decode TPS 为 9.99，约 1.27x。
- 完整 `mmlu_abstract_algebra` 100 条样本测试已完成：auto route 全部选择 PTQ，auto decode TPS 12.44，force FP decode TPS 9.91。
- `mmlu_abstract_algebra` 上 PTQ/auto 相对 force FP 的 token match mean 为 89.25%，exact sequence match 为 71%。
- batch sweep 已完成，batch size 覆盖 1/2/4/8/16。threshold 0.8 下 force PTQ 在 batch 1-8 相比 force FP 约 1.32x-1.38x；auto 因 batch-level all-pass gate 在大 batch 下基本退化为 FP。

## F. vLLM Baseline

- [ ] 官方 aarch64 image FP baseline 启动成功。
- [ ] benchmark harness 可记录 TTFT、TPOT、TPS、latency。
- [ ] 官方 image baseline 矩阵完成。
- [ ] source-based vLLM 源码 tag/commit 选择完成。
- [ ] source-built vLLM import 成功。
- [ ] source-built vLLM OpenAI server 启动成功。
- [ ] source-built FP baseline 与官方 image 性能接近。
- [ ] 完成 `experiments/05_vllm_baseline/official_image_summary.csv`。
- [ ] 完成 `experiments/06_source_vllm/source_selection.md`。

Go/No-Go：

- 如果 source-built baseline 明显慢于官方 image，先解决构建/依赖问题。

## G. vLLM Custom Model

- [ ] out-of-tree plugin 可被 vLLM 发现。
- [ ] custom full-FP mode 可启动 server。
- [ ] custom full-FP 输出正确。
- [ ] custom full-FP 性能 overhead 小于 10%。
- [ ] static FP tail mode 可运行。
- [ ] static PTQ tail mode 可运行。
- [ ] paged KV 多步 decode 不报错。
- [ ] 完成 `experiments/07_vllm_custom_model/summary.csv`。

Go/No-Go：

- custom full-FP overhead 不可接受时，不进入 routed execution。

## H. Naive Routed Execution

- [ ] `route_policy=all_fp` 可运行。
- [ ] `route_policy=all_ptq` 可运行。
- [ ] `route_policy=random` 可运行。
- [ ] `route_policy=river_gate` 可运行。
- [ ] 同步 routed merge 输出正确。
- [ ] 记录 route ratio。
- [ ] 记录 TPOT/TPS。
- [ ] 对比 static PTQ tail / full FP。
- [ ] 完成 `experiments/08_naive_routed/summary.csv`。

Go/No-Go：

- 如果 `all_ptq` 或 PTQ-heavy workload 无收益，不进入 microbatch runtime。

## I. Route-aware Microbatch Runtime

- [ ] 定义 `Q_pre_decode` item 格式。
- [ ] 定义 `Q_ptq_decode` item 格式。
- [ ] 定义 `Q_fp_decode` item 格式。
- [ ] PTQ microbatch 可独立 sampling。
- [ ] FP microbatch 可独立 sampling。
- [ ] PTQ 完成后不等待 FP。
- [ ] FP fairness 机制生效。
- [ ] 无 starvation。
- [ ] 输出 token 数正确。
- [ ] KV cache 管理稳定。
- [ ] 完成 `experiments/09_microbatch_runtime/summary.csv`。
- [ ] 完成 `experiments/09_microbatch_runtime/queue_trace.jsonl`。

通过标准：

- PTQ-heavy workload 下 mean TPOT 明显下降，P95/P99 不显著恶化。

## J. Scheduler 与 CUDA Graph

- [ ] 记录 phase-level latency。
- [ ] 记录 graph/eager 命中情况。
- [ ] 记录 batch size bucket。
- [ ] 记录 queue wait time。
- [ ] 实现 FCFS baseline。
- [ ] 实现 PTQ priority。
- [ ] 实现 PTQ priority + FP fairness。
- [ ] 对比 graph instrumentation 前后 TPOT。
- [ ] 完成 `experiments/10_scheduler_graph/summary.csv`。

通过标准：

- 能证明 scheduler 或 graph pool 对 wall-clock 指标有独立贡献。

## K. 论文实验材料

- [ ] 所有实验都有 `config.yaml`。
- [ ] 所有实验都有 `server.log`。
- [ ] 所有实验都有 `result.jsonl`。
- [ ] 所有实验都有 `summary.csv`。
- [ ] 所有实验都有 `notes.md`。
- [ ] 主表包含 TPS、TPOT、P95/P99、质量指标。
- [ ] 消融表包含 checkpoint k、route policy、scheduler、graph。
- [ ] 图包含 exit CDF、route ratio、queue wait、TPOT distribution。
- [ ] 整理 limitations。
- [ ] 整理 related work。

## 当前建议状态

当前项目应停留在 **E/F 之间**：

```text
PyTorch split reference 的 KV/质量 sanity
  -> vLLM baseline / custom model 准备
```

暂不建议进入：

```text
vLLM scheduler 改造
route-aware microbatch runtime
CUDA Graph pool 自定义
```

原因：

- 当前 checkpoint 已证明可以拆成 FP/PTQ 双 tail。
- PTQ tail 在 decode 型小矩阵上有效，但 prefill 长序列明显更慢。
- KV cache 兼容性与 fixed checkpoint 的跨任务质量稳定性还没有完成验证。
