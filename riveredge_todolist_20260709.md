# RiverEdge To-do List 2026-07-11

## 0. 当前研究路线

当前 RiverEdge 不再沿用原始 River 的“训练/early-exit”主线，而是聚焦：

```text
fixed checkpoint routed continuation
shared FP prefix -> route -> FP tail 或 PTQ tail
```

收益来源不是跳过层，而是让一部分请求在 decode 阶段走更快的 PTQ tail。当前采用 Llama-3.1-8B-Instruct + torchao serialized safetensors 路线，避免 HQQ wrapper 和运行时量化。

环境约定：

- 仓库根目录：`/home/orin/zjc/vllm`
- RiverEdge 目录：`/home/orin/zjc/vllm/RiverEdge`
- 模型目录：宿主机 `/media/orin/Data/models`，容器内 `/models`
- vLLM source 容器：`riveredge-vllm-src`
- vLLM 源码：`RiverEdge/river-vllm-edge/third_party/vllm`

## 1. Experiments 目录对齐

当前实际目录与阶段含义如下：

| 目录 | 状态 | 含义 |
|---|---|---|
| `experiments/03_weight_structure` | 已完成 | 旧 River/HQQ checkpoint 双 tail 可行性检查 |
| `experiments/04_split_reference` | 已完成/历史基线 | 旧 River/HQQ PyTorch split reference 与 lm-eval speed test |
| `experiments/05_vllm_baseline` | 已完成 | 官方 vLLM image baseline |
| `experiments/06_source_vllm` | 已完成 | source-based vLLM 环境与 smoke test |
| `experiments/07_vllm_custom_model` | 已完成/已淘汰 | 早期 vLLM custom model + HQQ wrapper 原型 |
| `experiments/08_llama_torchao_layer_quant` | 已完成/过渡 | Llama torchao online quant adapter 验证 |
| `experiments/09_torchao_serialized_riveredge` | 已完成/历史基线 | 分离式 torchao checkpoint、P4/P6/P7 实验 |
| `experiments/10_unified_fp_hqq_checkpoint` | 当前主线 | 单 checkpoint FP+HQQ、FP prefill、静态模式与多 k 验证 |

`00_problem_definition`、`01_env`、`02_river_original` 当前不再补建，避免与实际实验目录脱节。

## 2. 已完成：P3-P7

### P3：权重结构可行性

交付物：

- `experiments/03_weight_structure/model_structure_report.md`
- `experiments/03_weight_structure/module_tree.txt`
- `experiments/03_weight_structure/state_dict_keys.txt`

结论：旧 River checkpoint 可构造 `shared + FP tail` 和 `shared + PTQ tail`，但 HQQ 权重不能被 vLLM 原生高效读取。

### P4：PyTorch Reference

历史交付物：

- `experiments/04_split_reference/p4_summary.md`
- `experiments/04_split_reference/*batch*`
- `experiments/04_split_reference/*mmlu_abstract_algebra*`

新主线交付物：

- `river-vllm-edge/scripts/benchmark_torchao_fused_pytorch.py`
- `experiments/09_torchao_serialized_riveredge/p4_pytorch_fused_summary.csv`
- `experiments/10_unified_fp_hqq_checkpoint/p4_unified_smoke.json`

当前结论：torchao serialized PTQ tail 在 batch 1-4 有明显收益；batch 增大后优势收缩。

### P5：vLLM Baseline / Source Build

交付物：

- `experiments/05_vllm_baseline/official_image_summary.csv`
- `experiments/06_source_vllm/build_summary.md`
- `experiments/06_source_vllm/build_log.txt`

结论：source-based vLLM 环境可用，可作为后续修改 vLLM 的基础。

### P6：vLLM 静态模式

历史交付物：

- `experiments/07_vllm_custom_model/summary.md`

新主线交付物：

- `river-vllm-edge/scripts/export_riveredge_unified_checkpoint.py`
- `river-vllm-edge/scripts/benchmark_vllm_torchao_static_modes.py`
- `experiments/10_unified_fp_hqq_checkpoint/static_modes_k3.csv`

当前结论：统一 checkpoint 下 `static_ptq_tail` 单请求约 `19.95 tok/s`，FP 约 `11.06 tok/s`；prefill 明确走完整 FP，decode 才切换 HQQ tail。

### P7：Naive Routed

交付物：

- `river-vllm-edge/scripts/benchmark_vllm_naive_routed.py`
- `experiments/09_torchao_serialized_riveredge/p7_naive_routed_summary.csv`
- `experiments/09_torchao_serialized_riveredge/p7_naive_routed_routes.csv`

结论：engine 外部 naive routing 会拆 batch 并顺序执行 FP/PTQ batch，random 与 PTQ-heavy 都显著慢于 all-FP/all-PTQ。因此 naive dual-engine 不是最终路线。

## 3. 当前 Canonical Checkpoint

唯一保留的 canonical checkpoint：

- `/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`
- 完整 FP 第 1-32 层：`model.layers.0..31`
- HQQ 第 2-32 层：`model.ptq_layers.1..31`
- tensor 数据共 18.50 GiB，无符号链接或外部权重依赖

后续默认使用 k=3：

```text
layers 1-3: FP shared prefix
layers 4-32: PTQ tail candidate
layers 4-32: FP tail candidate
```

同一 checkpoint 已验证 `k=1,3,8,16,31`，不再为不同 k 生成权重。

## 4. P8：质量验证与质量校正速度

目标：确认 torchao PTQ checkpoint 不是只快但质量不可用。

任务：

- 比较统一 checkpoint 的 `full_fp` 与 `static_ptq_tail` lm-eval 准确率。
- 至少覆盖 `mmlu_abstract_algebra`、2-3 个 MMLU 子任务、一个生成任务 smoke。
- 记录 exact match / token match / perplexity-like sanity。
- 建立 quality-corrected speedup 表。

交付物：

- `experiments/11_quality_validation/summary.md`
- `experiments/11_quality_validation/accuracy_summary.csv`
- `experiments/11_quality_validation/sample_outputs.jsonl`

Go/No-Go：

- 如果 PTQ 质量下降不可接受，先调整量化配置或量化层范围，再进入 vLLM runtime 修改。

## 5. P9：解释 batch 下 PTQ 收益衰减

目标：用 profiler 解释为什么 PTQ 在 batch 增大后优势消失。

任务：

- 对 FP / PTQ 在 batch 1/2/4/8/16 下记录 prefill/decode TPS。
- 使用 nsys 或轻量 torch profiler 记录 kernel 时间。
- 分析 BF16 GEMM、torchao INT4 matmul、dequant/unpack、attention、norm 的占比。
- 明确当前 Orin + torchao kernel 的适用 batch 区间。

交付物：

- `experiments/12_ptq_batch_profile/summary.md`
- `experiments/12_ptq_batch_profile/tps_matrix.csv`
- `experiments/12_ptq_batch_profile/kernel_breakdown.csv`

## 6. P10：单 vLLM Model 内的 RiverEdge 双 tail

目标：停止使用双 engine，进入真正 RiverEdge 模型结构。

设计：

```text
shared FP layers 1-k
FP tail layers k+1-32
PTQ tail layers k+1-32
route decision
selected tail forward
```

已完成基础：

- `RiverEdgeUnifiedForCausalLM` 同时持有完整 FP 和 HQQ tail。
- 单 checkpoint loader、FP prefill、共享 paged KV cache 已跑通。
- 静态 `full_fp`、`static_fp_tail`、`static_ptq_tail` 已回归。
- 任意 `k>=1` 可直接选择 checkpoint 中对应 HQQ tail。

剩余任务：

- 在同一 forward 中加入按请求 route 的 FP/PTQ 选择。
- 对 mixed batch 做 logits 顺序恢复和正确性验证。
- 不修改 scheduler，先实现同步 routed tail baseline。

交付物：

- `river-vllm-edge/river_vllm_ext/models/unified_llama.py`
- `experiments/10_unified_fp_hqq_checkpoint/summary.md`
- `experiments/13_single_model_dual_tail/summary.md`
- `experiments/13_single_model_dual_tail/routed_summary.csv`

Go/No-Go：

- `all_fp` 与旧 native FP TPS 差异 < 2%，已满足。
- `all_ptq` 接近旧 P6 static PTQ-tail，已满足。
- mixed route 不应低于 P7 naive dual-engine。

## 7. P11：Route-aware Microbatch Runtime

目标：解决 naive routed 拆 batch 后顺序执行的问题。

任务：

- 在 vLLM scheduler/runner 层定义 route-aware batch。
- 将同一 step 内请求按 route 分组执行 tail。
- 合并 logits 并进入统一 sampler。
- 支持 PTQ-heavy workload 优先完成，不阻塞 FP-heavy workload。
- 记录 queue wait、route ratio、TPOT percentiles。

交付物：

- `experiments/14_route_aware_runtime/summary.md`
- `experiments/14_route_aware_runtime/queue_trace.jsonl`
- `experiments/14_route_aware_runtime/tpot_percentiles.csv`

## 8. P12：CUDA Graph / Compile / Scheduler 消融

目标：让 routed runtime 能利用固定 shape/bucket，降低调度和 Python overhead。

任务：

- 对比 eager、CUDA Graph、torch.compile。
- 记录 graph hit rate、fallback rate、batch bucket。
- 消融 FCFS、PTQ priority、PTQ priority + FP fairness。

交付物：

- `experiments/15_scheduler_graph/summary.md`
- `experiments/15_scheduler_graph/graph_trace.jsonl`
- `experiments/15_scheduler_graph/scheduler_ablation.csv`

## 9. 投稿前主表

需要最终形成：

- native vLLM FP
- static PTQ-tail
- single-model routed
- route-aware microbatch
- scheduler / CUDA Graph ablation
- quality-corrected speedup
- TPOT P50/P95/P99
- energy/token，如可测

交付物：

- `experiments/16_paper_results/main_table.csv`
- `experiments/16_paper_results/ablation_table.csv`
- `experiments/16_paper_results/figures/`
