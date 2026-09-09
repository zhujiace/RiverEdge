# RiverEdge PTQ Inductor 可行性与回退实验

## 目的

尝试在保留 PTQ full decode CUDA Graph 的前提下，将 RiverEdge PTQ tail 放入 vLLM 的 `torch.compile` 边界，并比较以下四种执行方式：

1. eager；
2. 仅 CUDA Graph；
3. 仅 Inductor；
4. Inductor + CUDA Graph。

环境为 Jetson AGX Orin、vLLM 0.24.0、PyTorch 2.11.0、BF16，权重为 `Llama-3.1-8B-Instruct-riveredge-fp-hqq`，`k=3`。

## Inductor 实现与结果

实验性实现将 FP shared layers 与 HQQ PTQ tail 移入带 `support_torch_compile` 的模型边界，prefill 仍绕过该边界执行全 FP。

- batch=1、16 output tokens 的 smoke test 可以完成：Inductor 为 22.42 tok/s，Inductor + CUDA Graph 为 23.01 tok/s，运行时均显示 `compiled=true`。
- TorchAO HQQ tensor subclass 缺少 `aten.empty_like` 实现，AOT compiled function 无法保存；因此即使本次运行成功，也不能稳定复用编译缓存。
- 动态编译范围在 batch=1 后切换至 batch=2 时失败：FlashAttention 收到错误的静态元数据，报错 `shape '[2, 8, 4, 128]' is invalid for input of size 4096`。
- 改用 `compile_sizes=[1,2,...]` 创建精确静态图仍失败：vLLM piecewise backend 报错 `Expected exactly one compiled range_entry ... but found 3`。

因此四组完整多 batch 消融无法安全完成。当前 vLLM 0.24 + TorchAO HQQ tensor subclass + 自定义 PTQ decode 边界不能可靠支持有效 batch 动态变化，不适合 RiverEdge continuous batching。

## 回退后验证

实验性模型改动已完全撤回，`unified_llama.py` 与仓库 `HEAD` 一致。基准脚本仅接受 eager 与 CUDA Graph，CUDA Graph 配置为 `CompilationMode.NONE + FULL_DECODE_ONLY`，不使用 Inductor。

配置：85 input tokens、32 output tokens、warmup 1 次、测量 3 次，PTQ tail，batch 1/2/4/8/16。

| Batch | eager TPS | CUDA Graph TPS | Graph 增益 |
|---:|---:|---:|---:|
| 1 | 20.13 | 24.68 | +22.57% |
| 2 | 37.04 | 46.80 | +26.32% |
| 4 | 68.63 | 77.92 | +13.54% |
| 8 | 91.71 | 93.28 | +1.71% |
| 16 | 121.90 | 123.43 | +1.25% |

所有 batch 和执行方式生成的 token 完全一致，phase 均为 `prefill_fp,decode_ptq`。五个 decode graph 捕获成功，图池实际占用约 0.07 GiB。

## 结论

后续默认使用 **PTQ + CUDA Graph，不使用 Inductor**。这是当前实现中吞吐最高且能覆盖 batch 1-16 的稳定配置。若未来重新研究 Inductor，应在独立分支中先解决 TorchAO tensor subclass AOT 序列化和 vLLM 多静态范围管理，不能直接合入现有推理路径。

结果见 `post_revert_validation.json`、`post_revert_validation.csv` 和 `post_revert_validation.log`；失败证据保存在 `ptq_four_way_matrix.log` 与 `inductor_batch12_smoke.log`。
