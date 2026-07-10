# TorchAO Safetensors RiverEdge 实验汇总

## 目标

将 Llama-3.1-8B-Instruct 的部分层导出为 torchao HQQ INT4 weight-only safetensors checkpoint，使 vLLM 可以直接读取已量化权重；随后基于该权重重新测试 P4、P6、P7。

## Checkpoint

基础权重：

- `/models/Llama-3.1-8B-Instruct`
- 原始目录大小：`22G`

已生成 checkpoint：

| 路径 | 用途 | 量化层 | 量化 fused Linear 数 | 大小 |
|---|---:|---:|---:|---:|
| `/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-fused` | 全面层覆盖测试 | 2-32 | 124 | 6.0G |
| `/models/Llama-3.1-8B-Instruct-layer4-32-torchao-hqq-fused` | k=3 static PTQ-tail | 4-32 | 116 | 6.6G |

实现脚本：

- `RiverEdge/river-vllm-edge/scripts/export_llama_torchao_fused_checkpoint.py`
- 将 HF split 权重拼成 vLLM fused layout：`qkv_proj`、`gate_up_proj`
- 使用 torchao `Int4WeightOnlyConfig(group_size=64, HQQ, TILE_PACKED_TO_4D, ntile=8)`
- 使用 `flatten_tensor_state_dict` 写入 safetensors metadata
- 在 `config.json` 中嵌入 `quantization_config.quant_method=torchao`

验证结果：

- vLLM 可直接识别 `quantization=torchao`
- layer2-32 checkpoint smoke 推理成功
- 权重加载约 `2.01s`，模型显存约 `5.97GiB`

## P4：PyTorch Fused Proxy

脚本：

- `RiverEdge/river-vllm-edge/scripts/benchmark_torchao_fused_pytorch.py`

说明：该测试不使用 vLLM，也不做运行时量化。它直接从 safetensors 恢复 torchao tensor subclass，用真实 fused 权重跑 Llama-like per-token linear stack。该测试不建模 attention score 和 KV cache，主要用于观察 INT4 fused Linear 在 PyTorch/torchao 下的速度潜力。

结果：

| mode | b1 tok/s | b2 tok/s | b4 tok/s | b8 tok/s | b16 tok/s |
|---|---:|---:|---:|---:|---:|
| full_fp | 11.53 | 23.07 | 45.80 | 91.23 | 182.46 |
| shared_fp_plus_fp_tail | 11.54 | 23.00 | 45.86 | 90.98 | 181.93 |
| shared_fp_plus_ptq_tail | 21.71 | 43.05 | 85.77 | 135.13 | 170.49 |
| all_ptq_tail | 21.42 | 42.61 | 84.85 | 139.35 | 169.59 |

现象：

- batch 1-4 下，PTQ tail 相比 FP 约 `1.87x-1.88x`。
- batch 8 下收益降到约 `1.48x`。
- batch 16 下 PTQ 不再更快，说明当前 torchao INT4 kernel 对较大 batch 的优势会减弱。

## P6：vLLM 三种静态模式

脚本：

- `RiverEdge/river-vllm-edge/scripts/benchmark_vllm_torchao_static_modes.py`

设置：

- `max_model_len=128`
- `max_new_tokens=16`
- 单请求
- `enforce_eager=True`
- `static_fp_tail` 当前与 full-FP 计算图等价，仅作为静态模式基线

结果：

| mode | model | TPS | mean latency |
|---|---|---:|---:|
| full_fp | FP base | 11.23 | 1.425s |
| static_fp_tail | FP base | 11.23 | 1.425s |
| static_ptq_tail | layer4-32 PTQ | 20.44 | 0.783s |

结论：

- vLLM 直接读取 serialized torchao 权重后，k=3 static PTQ-tail 单请求 TPS 约为 FP 的 `1.82x`。
- 这说明“已量化权重 + vLLM torchao 原生路径”确实能发挥 PTQ tail 的速度优势。

## P7：Naive Routed

脚本：

- `RiverEdge/river-vllm-edge/scripts/benchmark_vllm_naive_routed.py`

设置：

- `num_prompts=8`
- `max_new_tokens=16`
- all-FP/all-PTQ：单 engine，`gpu_memory_utilization=0.75`
- mixed route：FP engine + PTQ engine，顺序执行两个 batch，`gpu_memory_utilization=0.32`

结果：

| policy | FP req | PTQ req | TPS | total time |
|---|---:|---:|---:|---:|
| all_fp | 8 | 0 | 87.16 | 1.468s |
| all_ptq | 0 | 8 | 94.50 | 1.355s |
| random | 5 | 3 | 54.80 | 2.336s |
| ptq_heavy | 1 | 7 | 48.15 | 2.658s |

现象：

- batch=8 时 all-PTQ 只比 all-FP 快约 `8.4%`，远低于单请求 P6 的 `1.82x`。
- naive mixed route 明显变慢。原因是请求被拆成 FP batch 和 PTQ batch 顺序执行，破坏了 vLLM continuous batching 的合批收益。
- 即使 PTQ-heavy 中 7/8 请求走 PTQ，只要有 1 个 FP 请求，也要额外执行一次 FP batch，因此 TPS 低于 random。

## 结论

1. torchao safetensors serialized checkpoint 路线可行，vLLM 能直接读取并执行，不需要在线量化或 HQQ wrapper。
2. 对单请求或小 batch，PTQ tail 有明确收益；P4/P6 均显示约 `1.8x` 量级加速。
3. batch 增大后 PTQ 优势明显收缩，batch=8 的 vLLM all-PTQ 只比 all-FP 快约 `8.4%`。
4. naive routed 双 engine 不适合作为最终方案；它会拆 batch、顺序执行、损失 continuous batching 效率。
5. 后续优化重点应放在 vLLM 内部 route-aware batching：共享 1-k 层、按 route 分组 tail、同一 scheduler step 内合并结果，而不是在 engine 外部做请求级分流。

