# Llama 3.1 8B Layer 2-32 TorchAO/HQQ vLLM Adapter

## 目标

为 RiverEdge 后续开发准备一个 vLLM 原生可加载的 Llama 3.1 8B 量化基础：第 1 层保持 FP，2-32 层进行 INT4 HQQ weight-only 量化。这里的 2-32 是 1-indexed，对应 vLLM/HF layer index `1..31`。

## 实现方式

- 基础权重：`/models/Llama-3.1-8B-Instruct`
- Adapter：`/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-online`
- 量化配置：`/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-online/torchao_layer_quant_config.json`
- vLLM 量化方式：`quantization="torchao"` + `hf_overrides={"quantization_config_file": ...}`
- torchao：`0.17.0`
- 量化算法：`Int4WeightOnlyConfig(group_size=64, HQQ, TILE_PACKED_TO_4D)`

没有把 `quantization_config` 写入 `config.json`。原因是 vLLM 会把 HF `config.json` 中的 torchao `quantization_config` 解释为“已序列化 torchao checkpoint”；本实验需要的是从 FP checkpoint 加载后由 vLLM 在线量化。

## 量化范围

每层量化 4 个 vLLM fused Linear：

- `self_attn.qkv_proj`
- `self_attn.o_proj`
- `mlp.gate_up_proj`
- `mlp.down_proj`

层范围为 `model.layers.1..31`，共 `31 * 4 = 124` 个模块。`model.layers.0`、embedding、norm、lm_head 保持 FP。

## 脚本

- 生成 adapter：`RiverEdge/river-vllm-edge/scripts/create_llama_torchao_layer_quant_adapter.py`
- smoke benchmark：`RiverEdge/river-vllm-edge/scripts/benchmark_llama_torchao_layer_quant.py`

生成命令：

```bash
cd /workspace/vllm/RiverEdge/river-vllm-edge
python3 scripts/create_llama_torchao_layer_quant_adapter.py \
  --source /models/Llama-3.1-8B-Instruct \
  --output /models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-online \
  --layer-start 2 \
  --layer-end 32 \
  --group-size 64 \
  --force
```

## Smoke Test

环境：`riveredge-vllm-src` 容器，vLLM source `0.24.0`，`max_model_len=128`，`max_new_tokens=8`，`enforce_eager=True`。

| mode | load_s | model memory | output TPS |
|---|---:|---:|---:|
| FP | 56.06 | 15.0 GiB | 10.82 |
| layer2-32 INT4 HQQ | 105.12 | 5.97 GiB | 18.60 |

结论：vLLM 能正常加载并推理；量化路径显著降低模型权重显存占用，并在该短生成 smoke test 中高于 FP TPS。加载时间更长，因为当前是在线量化 FP checkpoint。

## 注意

当前容器 PyTorch 会提示 Orin CC 8.7 不在该 wheel 的显式 build list 中，但 torchao INT4 小 Linear 和 vLLM Llama smoke test 均已通过。
