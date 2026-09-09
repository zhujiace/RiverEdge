# 单请求端到端效率剖析

## 结论摘要

- 配置：Llama-3.1-8B-Instruct、RiverEdge `k=3`、85 input tokens、32 output tokens、batch/concurrency=1、BF16、FULL_DECODE_ONLY CUDA Graph、Inductor 关闭。
- FP：TTFT `115.27 ms`，稳态 TPOT `87.83 ms/token`，端到端 `2838.25 ms`，输出吞吐 `11.27 tok/s`。
- PTQ tail：TTFT `115.07 ms`，稳态 TPOT `38.02 ms/token`，端到端 `1293.89 ms`，输出吞吐 `24.73 tok/s`。
- PTQ 将 TPOT 降低 `56.7%`（`2.31x`），端到端输出 TPS 提高 `2.19x`。TPS 提升略小，因为约 115 ms 的全 FP prefill/TTFT 没有变化。
- 结论：当前单请求瓶颈不是调度或采样，而是 decode 每步读取并执行大规模线性层权重；PTQ 正在优化正确的主要路径。

## 关键路径

| Decode 阶段 | FP (ms/token) | PTQ tail (ms/token) | PTQ 占 TPOT |
|---|---:|---:|---:|
| Model forward | 81.793 | 32.037 | 84.3% |
| LM-head / logits | 5.878 | 5.830 | 15.3% |
| 其余关键路径 | 0.164 | 0.158 | 0.4% |

CUDA Graph 下，FP model forward 从 `81.79` 降到 PTQ 的 `32.04 ms/token`。LM head 仍为约 `5.83 ms/token`，从 FP 中的 `6.7%` 上升到 PTQ 中的 `15.3%`，成为下一阶段明确的优化对象。

CPU 的 `output.wait_gpu_to_cpu` 数字表示阻塞等待 GPU，不是 CPU 在做同等时长的计算。调度、状态更新、detokenize 等 CPU 工作均远小于 1 ms/step；GPU 输入准备约 0.06 ms/step，sampler 约 0.08 ms/step。

| 细分阶段 | FP (ms) | PTQ tail (ms) | 解释 |
|---|---:|---:|---|
| Prefill model forward | 105.711 | 105.585 | 85-token prefill，全 FP |
| Prefill logits | 5.889 | 5.827 | 首 token LM head |
| Scheduler / step | 0.101 | 0.100 | CPU 决策 |
| Prepare inputs / step | 1.552 | 1.606 | CPU wall，包含若干子阶段 |
| GPU-to-CPU wait / step | 83.465 | 35.136 | 与 GPU 执行重叠，不可再次相加 |
| Detokenize / token | 0.066 | 0.068 | CPU 输出处理 |

`critical_path.csv` 继续列出 state update、shape selection、slot mapping、attention metadata、model launch、sampling 和 scheduler update。CPU parent range、子 range及 GPU range存在嵌套或异步重叠，不能直接全部求和；可相加的 GPU 主路径已在上表“关键路径”中单独给出。

## Kernel 归因

- FP Nsys：总 kernel 时间 `2868.0 ms/请求`，其中 BF16 GEMM `2821.0 ms`（`98.4%`）。
- PTQ Nsys：总 kernel 时间 `1319.1 ms/请求`；INT4 tinygemm `724.4 ms`，BF16 GEMM `550.0 ms`，二者合计 `96.6%`。
- PTQ eager NVTX 的 kernel 归因估算：每个 decode step 的 4-32 层 PTQ tail 约 `27.20 ms`，1-3 层 shared FP 约 `7.98 ms`；PTQ tail 内 attention 约 `6.08 ms`，MLP 约 `20.75 ms`。这些是 kernel 累计时间，不包含 CPU launch gap。
- PTQ 的 Device-to-Device 拷贝虽增加到 4683 次，但整请求仅约 5.8 ms；它不是当前 1.29 s 端到端时间的主因。

## CUDA Graph

- FP eager TPOT `88.70 ms`，CUDA Graph `87.83 ms`，仅改善 `1.0%`。FP GEMM 很长，kernel launch 开销占比较低。
- PTQ eager TPOT `48.15 ms`，CUDA Graph `38.02 ms`，改善 `21.0%`。量化后 kernel 更短且更碎，CUDA Graph 的价值明显增大。

## 优化优先级

1. **LM head / logits**：约 `5.83 ms/token`，占 PTQ TPOT `15.3%`。研究量化 LM head、低比特 vocab projection，或在保持语义的前提下优化 top-1 logits 路径。
2. **INT4 tail kernel**：验证 tinygemm 在 Orin `M=1` 下的内存带宽、tensor-core 利用率和反量化融合；定制 kernel 应优先融合 scale/zero-point、GEMM、bias/activation，减少中间写回。
3. **Shared FP 1-3 层**：约 `7.98 ms/token` 的不可退出固定成本。评估减小 `k`、只保留必要 shared state，或对 shared 层采用更温和的量化。
4. **保持 full decode CUDA Graph**：PTQ 下 Graph 已贡献约 `21.0%` TPOT 改善。动态 route 应按有限的静态路径/桶 capture，避免逐 token graph break。
5. **Prefill 单独优化**：当前按设计全 FP，所以 TTFT 没有改善。若目标包含 TTFT，再研究 prefill 权重量化或 chunked prefill；不要与 decode TPOT 结论混合。

## 测量边界

主结果是 vLLM 进程内从 raw prompt 请求接收、tokenize、schedule、GPU 执行、sample、detokenize 到最终输出的端到端时间，刻意排除了 loopback HTTP/SSE，以免网络栈掩盖模型关键路径。Nsys 只用于 kernel 构成，不能把 profiler 下 wall time当作正常性能。

CUDA Event runtime 插桩对 CUDA Graph E2E 的扰动低于 0.1%。eager 层级 hook 对 FP 的扰动为 `0.4%`，对 PTQ 为 `34.5%`；因此 PTQ 层级 JSON 的绝对 wall time不作性能结论，层级归因采用更轻的 Nsys NVTX kernel 统计。

完整数据见 `request_summary.csv`、`critical_path.csv`、`runtime_stage_summary.csv`、`token_timeline.csv`、`engine_steps.csv`、`layer_event_summary.csv`、`nsys_kernel_categories.csv`、`nsys_layer_kernel_summary.csv` 和 `nsys/` 下原始 trace。
