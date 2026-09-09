# RiverEdge vLLM Batch 与 CUDA Graph 实验

## 实验目的

验证 unified RiverEdge 是否能够直接执行 batch 推理，并比较 eager 与 CUDA Graph 下 FP、HQQ PTQ tail 的吞吐。

## 配置

- 平台：NVIDIA Jetson AGX Orin，vLLM 0.24.0，PyTorch 2.11.0，BF16
- 权重：`Llama-3.1-8B-Instruct-riveredge-fp-hqq`
- RiverEdge：`k=3`；prefill 全 FP；decode 使用全 FP 或 HQQ layer 4-32
- 请求：85 input tokens，32 output tokens，`ignore_eos=True`
- Batch：1、2、4、8、16；每点 warmup 1 次、测量 3 次
- Prefix cache、chunked prefill、Inductor 均关闭
- CUDA Graph：`FULL_DECODE_ONLY`，仅捕获 1、2、4、8、16 五种尺寸
- `gpu_memory_utilization=0.50`，图池实际增加约 0.07 GiB

## TPS 结果

TPS 为一次 `LLM.generate()` 中全部请求的 output token throughput，计时包含 FP prefill、decode 和 vLLM 调度。

| Batch | FP eager | PTQ eager | PTQ/FP | FP Graph | PTQ Graph | PTQ/FP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.18 | 19.92 | 1.78x | 11.30 | 24.65 | 2.18x |
| 2 | 21.81 | 36.87 | 1.69x | 22.12 | 46.85 | 2.12x |
| 4 | 41.54 | 67.87 | 1.63x | 42.07 | 77.91 | 1.85x |
| 8 | 71.10 | 91.68 | 1.29x | 71.91 | 93.23 | 1.30x |
| 16 | 129.15 | 121.88 | 0.94x | 130.66 | 123.37 | 0.94x |

## CUDA Graph 增益

| Batch | FP Graph/Eager | PTQ Graph/Eager |
|---:|---:|---:|
| 1 | +1.11% | +23.73% |
| 2 | +1.43% | +27.06% |
| 4 | +1.28% | +14.80% |
| 8 | +1.15% | +1.68% |
| 16 | +1.17% | +1.23% |

所有模式生成的 32 个 token 完全一致。PTQ 的 `phase_trace` 均为 `prefill_fp,decode_ptq`，CUDA Graph 捕获没有改变 RiverEdge phase。

## 结论

1. 当前 unified RiverEdge 可以直接执行静态 batch。等长请求同时进入调度器后，prefill 使用全 FP，纯 decode batch 能整体进入 PTQ tail。
2. PTQ 并非在所有 batch 下失效。它在 batch 1-8 仍快于 FP，但优势持续下降，并在 batch=16 反转；当前端到端交叉点位于 8 与 16 之间。
3. CUDA Graph 对 FP 仅提升约 1%，但对 batch 1-4 的 PTQ 提升 15%-27%。HQQ 小 batch 路径包含较多短 kernel，减少 launch 开销具有明显收益；大 batch 主要受 kernel/GEMM 本身限制。
4. 当前最佳单请求配置为 PTQ tail + CUDA Graph，达到 24.65 tok/s，比 FP eager 提高 2.20x。
5. 本实验不是 mixed-route continuous batching。同一个 forward batch 只能整体执行 FP 或 PTQ；逐请求路由、压缩/恢复及异步到达仍需 scheduler 集成。

## 完整 GSM8K 补充实验

使用完整 GSM8K test 1319 条、Llama Instruct 8-shot chat prompt、自然 EOS 和封闭 batch=16 进行补充测试。Full FP 为 50.57 tok/s、82.71% strict accuracy；static PTQ tail 为 59.58 tok/s、82.41% strict accuracy。PTQ 的 output TPS 提升 17.83%，完成时间缩短 13.83%，准确率差异不显著（McNemar exact `p=0.800`）。

真实变长请求会使活跃 decode batch 从 16 逐步下降，因此 PTQ 能重新利用 batch 1-8 的优势；但封闭 batch drain 也使两种模式都显著低于固定 32-token 满 batch 的 TPS。完整配置、结果和逐样本配对见 `gsm8k_full/summary.md`。

## 结果文件

- `eager_matrix.json`、`eager_matrix.csv`
- `cudagraph_matrix.json`、`cudagraph_matrix.csv`
- `cudagraph_matrix.log`
- `gsm8k_full/summary.md` 及该目录下完整结果
- 基准脚本：`river-vllm-edge/scripts/benchmark_vllm_unified_batch_cudagraph.py`
- GSM8K 脚本：`river-vllm-edge/scripts/benchmark_vllm_gsm8k.py`
