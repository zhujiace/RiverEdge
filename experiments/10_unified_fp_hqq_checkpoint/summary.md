# 统一 FP+HQQ Checkpoint 验证

## 目标

将完整 Llama-3.1-8B BF16 权重和第 2-32 层 TorchAO HQQ INT4 权重组织为一个 checkpoint，支持任意 `k>=1`，并保证 RiverEdge prefill 始终使用完整 FP 模型。

## Checkpoint

- 路径：`/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`
- FP：第 1-32 层，16,060,522,496 bytes
- HQQ：第 2-32 层的 QKV/O/GateUp/Down，3,803,185,152 bytes
- 总 tensor 数据：19,863,707,648 bytes（18.50 GiB）
- vLLM 实际模型权重占用：18.58 GiB
- HQQ 模块：31 层、124 个 fused Linear
- 目录无符号链接，不依赖旧 base/fused checkpoint。
- 验证后已删除原始 base、online adapter、layer2-32 fused 和 layer4-32 fused 目录；清理后重新加载并生成成功。

权重命名：

```text
model.layers.0..31       # 完整 FP 模型
model.ptq_layers.1..31   # HQQ 线性层
```

PTQ tail 复用 FP 层的 RMSNorm、RoPE、Attention operator 和 paged KV cache，只替换线性投影。可判定为纯 decode 的 batch 才启用 PTQ；prefill 和含多-token prefill 的 mixed batch 使用 FP。

当前 phase 判定已覆盖静态单请求（包括单 token 初始 prompt）；动态连续批处理中逐请求 mixed route 仍需 P11 scheduler 提供显式请求级 phase/route metadata。

## 静态模式回归

设置：Orin、vLLM 0.24.0、BF16、eager、`k=3`、prompt 约 64 tokens、输出 16 tokens、warmup 1、重复 3 次、关闭 prefix cache。

| 模式 | Prefill | Decode | TPS | 平均延迟 |
|---|---|---|---:|---:|
| full_fp | FP 1-32 | FP 1-32 | 11.06 | 1.447 s |
| static_fp_tail | FP 1-32 | FP 1-32 | 11.07 | 1.446 s |
| static_ptq_tail | FP 1-32 | FP 1-3 + HQQ 4-32 | 19.95 | 0.802 s |

静态 PTQ 相对 full FP 为 **1.80x**。三个模式生成的 16 个 token 完全一致。`phase_trace` 分别验证了 `all_fp` 和 `prefill_fp + decode_ptq`。

## 多 k 验证

同一个模型实例依次测试 `k=1,3,8,16,31`：

- 所有 k 均成功执行，无需重新生成或切换 checkpoint。
- 所有 k 均记录到 `prefill_fp` 和 `decode_ptq`。
- 所有 k 的首 token 与 full FP 一致。
- 结果见 `multi_k_validation.json`。
- 清理后的独立加载结果见 `post_cleanup_smoke.json`。

## 兼容性

- P4 PyTorch fused-linear proxy 已改为通过 `--checkpoint` 读取统一权重，并完成 smoke test。
- P6 三种 vLLM 静态模式已改为通过 `hf_overrides` 选择同一 checkpoint 中的模式。
- 旧 P7 dual-engine routed 仅保留为历史反例；后续 route 应直接在该单模型中实现。

注意：无 warmup 的首次 PTQ decode 会触发 TorchAO kernel/JIT，冷启动结果不代表稳态 TPS。
