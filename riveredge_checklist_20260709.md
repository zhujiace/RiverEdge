# RiverEdge Project Checklist 2026-07-11

该 checklist 与当前 `experiments/` 实际目录对齐。当前主线是 Llama-3.1-8B + torchao serialized safetensors + vLLM routed continuation。

## A. 当前路线

- [x] 明确路线：adaptive-precision routed continuation，不是训练型 early exit。
- [x] 明确收益来源：PTQ tail decode 加速，而不是跳过层。
- [x] 确认 HQQ wrapper 路线不适合 vLLM 原生高效推理。
- [x] 确认 torchao serialized safetensors 是当前主线。
- [x] 当前报告：`experiments/09_torchao_serialized_riveredge/summary.md`。

通过标准：

- 后续实验都围绕 `shared FP prefix + FP/PTQ tail route` 组织。

## B. Experiments 目录状态

- [x] `experiments/03_weight_structure`：旧 River/HQQ 权重结构检查完成。
- [x] `experiments/04_split_reference`：旧 PyTorch split reference 与 dataset speed test 完成。
- [x] `experiments/05_vllm_baseline`：官方 vLLM image baseline 已记录。
- [x] `experiments/06_source_vllm`：source-based vLLM 环境完成。
- [x] `experiments/07_vllm_custom_model`：早期 HQQ wrapper vLLM 原型完成，已标记为历史基线。
- [x] `experiments/08_llama_torchao_layer_quant`：torchao online quant adapter 验证完成，已被 serialized checkpoint 替代。
- [x] `experiments/09_torchao_serialized_riveredge`：当前主线结果完成。

说明：

- 不再强制补建 `00_problem_definition`、`01_env`、`02_river_original`。
- 后续新目录从 `10_quality_validation` 开始。

## C. Checkpoint 与权重

- [x] 下载并验证 `/models/Llama-3.1-8B-Instruct`。
- [x] 生成 `/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-fused`。
- [x] 生成 `/models/Llama-3.1-8B-Instruct-layer4-32-torchao-hqq-fused`。
- [x] vLLM 可直接从 config.json 识别 `quantization=torchao`。
- [x] vLLM 直接加载 serialized checkpoint 推理成功。
- [x] 记录 checkpoint 大小与量化层范围。
- [ ] 为未来不同 k 生成 checkpoint 配置表。

通过标准：

- 不依赖 HQQ wrapper，不进行运行时量化，vLLM 直接读取已量化权重。

## D. P4 PyTorch Reference

- [x] 旧 River/HQQ split reference 已完成。
- [x] 旧 lm-eval dataset speed test 已完成。
- [x] 新 torchao fused PyTorch proxy 已完成。
- [x] 记录 batch 1/2/4/8/16 下 FP/PTQ TPS。
- [x] 发现 batch 增大后 PTQ 优势衰减。
- [ ] 补完整语义生成路径的 PyTorch dual-tail quality sanity。

当前关键结果：

| mode | b1 tok/s | b4 tok/s | b8 tok/s | b16 tok/s |
|---|---:|---:|---:|---:|
| full_fp | 11.53 | 45.80 | 91.23 | 182.46 |
| shared_fp_plus_ptq_tail | 21.71 | 85.77 | 135.13 | 170.49 |

通过标准：

- PTQ tail 在小 batch decode 场景有明确收益，但不能假设大 batch 下仍有收益。

## E. P5 vLLM Baseline / Source Build

- [x] 官方 image baseline 记录完成。
- [x] source vLLM 路径确定。
- [x] source vLLM import 成功。
- [x] source vLLM smoke server / completion 验证完成。
- [x] 记录 build log 与 source selection。
- [ ] 后续新模型实验统一记录 vLLM commit/version、torchao version、dtype、CUDA Graph 设置。

通过标准：

- 当前环境足够支持 Python-level vLLM custom model 开发。

## F. P6 vLLM 静态模式

- [x] 旧 `experiments/07_vllm_custom_model` 三模式 smoke 完成。
- [x] 新 serialized static PTQ-tail vLLM benchmark 完成。
- [x] `full_fp` 单请求 TPS 已记录。
- [x] `static_fp_tail` 单请求 TPS 已记录。
- [x] `static_ptq_tail` 单请求 TPS 已记录。
- [x] 证明 serialized PTQ-tail 在 vLLM 单请求下有效。
- [ ] 补 CUDA Graph 非 eager 对比。
- [ ] 补更长 decode token 的稳定 TPS。

当前关键结果：

| mode | TPS |
|---|---:|
| full_fp | 11.23 |
| static_fp_tail | 11.23 |
| static_ptq_tail | 20.44 |

通过标准：

- static PTQ-tail 相比 FP 单请求有明确收益，已通过。

## G. P7 Naive Routed

- [x] `all_fp` policy 已完成。
- [x] `all_ptq` policy 已完成。
- [x] `random` policy 已完成。
- [x] `ptq_heavy` mixed policy 已完成。
- [x] route ratio 已记录。
- [x] TPS 已记录。
- [x] 确认 naive dual-engine mixed route 会显著降速。
- [ ] 不再将 dual-engine naive routed 作为最终系统方案。

当前关键结果：

| policy | FP | PTQ | TPS |
|---|---:|---:|---:|
| all_fp | 8 | 0 | 87.16 |
| all_ptq | 0 | 8 | 94.50 |
| random | 5 | 3 | 54.80 |
| ptq_heavy | 1 | 7 | 48.15 |

通过标准：

- naive route 只作为反例基线；后续必须进入单模型或 scheduler 内 route-aware runtime。

## H. P8 质量验证

- [ ] 创建 `experiments/10_quality_validation/`。
- [ ] FP base lm-eval baseline。
- [ ] layer4-32 PTQ checkpoint lm-eval。
- [ ] 至少 3 个 MMLU 子任务。
- [ ] 一个生成任务 smoke。
- [ ] 记录 accuracy / exact match / token match。
- [ ] 输出 quality-corrected speedup。

Go/No-Go：

- 如果质量下降不可接受，先调整量化层范围、group size 或保留更多 FP 层。

## I. P9 PTQ Batch Profile

- [ ] 创建 `experiments/11_ptq_batch_profile/`。
- [ ] batch 1/2/4/8/16 TPS matrix。
- [ ] 区分 prefill 与 decode。
- [ ] 记录 kernel-level breakdown。
- [ ] 分析 BF16 GEMM 与 torchao INT4 matmul 的利用率。
- [ ] 解释 PTQ 优势消失的 batch 阈值。

通过标准：

- 能用 profiler 数据解释“PTQ 小 batch 快、大 batch 不快”。

## J. P10 单模型双 Tail vLLM

- [ ] 创建 `experiments/12_single_model_dual_tail/`。
- [ ] 设计 `RiverEdgeLlama`，同时持有 FP tail 与 PTQ tail。
- [ ] 解决 base FP checkpoint 与 PTQ safetensors 双源加载。
- [ ] `all_fp` 静态模式可运行。
- [ ] `all_ptq` 静态模式可运行。
- [ ] mixed route forward 可运行。
- [ ] 输出 logits merge 正确。
- [ ] paged KV 多步 decode 不报错。
- [ ] 对比 P6/P7 TPS。

Go/No-Go：

- mixed route 必须快于 P7 naive dual-engine，否则不进入 scheduler 改造。

## K. P11 Route-aware Runtime

- [ ] 创建 `experiments/13_route_aware_runtime/`。
- [ ] 定义 route-aware scheduler item。
- [ ] 同 step 内按 route 分组 tail batch。
- [ ] PTQ tail 与 FP tail logits merge。
- [ ] sampler 输出顺序正确。
- [ ] 记录 queue wait。
- [ ] 记录 TPOT P50/P95/P99。
- [ ] 记录 starvation/fairness。

通过标准：

- PTQ-heavy workload 下 mean TPOT 明显下降，P95/P99 不显著恶化。

## L. P12 Scheduler / CUDA Graph 消融

- [ ] 创建 `experiments/14_scheduler_graph/`。
- [ ] eager vs CUDA Graph 对比。
- [ ] graph hit rate 记录。
- [ ] batch bucket 记录。
- [ ] FCFS baseline。
- [ ] PTQ priority。
- [ ] PTQ priority + FP fairness。
- [ ] 输出 scheduler ablation。

通过标准：

- 能证明 scheduler 或 graph pool 对 wall-clock 指标有独立贡献。

## M. 论文结果

- [ ] 创建 `experiments/15_paper_results/`。
- [ ] 主表：TPS、TPOT、P50/P95/P99、quality。
- [ ] 消融：k、route policy、batch size、scheduler、CUDA Graph。
- [ ] 图：route ratio、TPOT distribution、queue wait、kernel breakdown。
- [ ] limitations。
- [ ] related work 对照。

## 当前建议状态

当前项目应处于：

```text
P8 quality validation
P9 PTQ batch profiling
```

然后再进入：

```text
P10 single-model dual-tail vLLM
P11 route-aware runtime
P12 scheduler/CUDA Graph
```

暂不建议继续扩展 dual-engine naive routed，因为实验已经证明它会破坏 batching 收益。
