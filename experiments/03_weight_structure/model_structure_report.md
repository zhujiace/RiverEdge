# P3 权重结构检查报告

## 结论

`Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16` 可以支持 PyTorch 原型层面的“双 tail”：

- `shared FP prefix + FP tail`：使用原始 `model.layers[k:]`。
- `shared FP prefix + PTQ tail`：使用 `model.exit_modules[0][k:]` 中的 HQQ 量化 decoder layer。

因此 P3 通过，可以进入 P4。需要注意：该 checkpoint 没有量化第 0 层，所以不存在从 embedding 后直接全 PTQ 的 32 层路径；最早只能是 `layer0 FP + layer1..31 PTQ`。

## 结构证据

模型配置：

- `model_type=sideway_llama`
- `num_hidden_layers=32`
- `exit_arch=river`
- `exit_decoder_layer=True`
- `exit_layer_indices=1..31`
- `output_exit_layers=1..31`
- `quantize_exit_layers=True`
- `quantize_full_model=False`
- `tie_exit_lm_head=True`

checkpoint key 统计：

- 总 key 数：4507
- FP backbone：`model.layers.0..31`，共 288 个参数 key
- PTQ exit/tail：`model.exit_modules.*`，共 4216 个 key
- `model.exit_modules.0`：index `1..31`，31 个 HQQ 量化 decoder layer
- `model.exit_modules.1`：index `0..30`，31 个 exit RMSNorm

代码路径：

- 初始化时，River 架构构造 `exit_modules[0] = [Identity]*start_index + DecoderLayer(start_index..31)`。
- forward 时，某个退出层 `k` 会执行 `flow = self.exit_modules[0][exit_index+1:]`，即从主干第 `k` 层之后接量化 tail。
- tail 后接 `self.exit_modules[1][exit_index]` 的 RMSNorm，再共用 `lm_head`。

## 约束

- 当前 PTQ tail 是一条共享的量化层序列，不是每个 exit layer 单独一份 tail。
- 当前 P3 只确认 no-cache/PyTorch forward 结构可用；完整 KV cache 语义还需要后续 runtime 验证。
- PTQ tail 适合作为 decode 阶段候选路径；prefill 长序列是否使用 PTQ tail 需要由 P4 性能决定。
