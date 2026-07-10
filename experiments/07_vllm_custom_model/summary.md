# RiverEdge vLLM Custom Model 初步测试

日期：2026-07-10

## 环境

- 容器：`riveredge-vllm-src`
- vLLM 源码：`/workspace/vllm/RiverEdge/river-vllm-edge/third_party/vllm`
- vLLM 版本：`0.24.0`
- 权重：`/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16`
- 默认退出层：`k=3`
- 必要环境变量：`VLLM_ENABLE_V1_MULTIPROCESSING=0`

## 已实现内容

- 新增 vLLM plugin：`river_vllm_ext.plugin:register`
- 注册 `RiverEdgeLlamaForCausalLM`：用于 `full_fp` 和 `static_fp_tail`
- 注册 `RiverEdgeHFStaticPTQForCausalLM`：用于单请求 `static_ptq_tail` 原型
- 新增 adapter 生成脚本，将 River checkpoint 包装成 vLLM 可识别的 Llama config
- 新增 benchmark 脚本，每个模式单独子进程运行，避免 NCCL/vLLM 状态残留

## 当前实现语义

- `full_fp`：vLLM 原生 Llama 全 FP 路径。
- `static_fp_tail`：当前阶段计算图与 `full_fp` 等价，只保留模式标记；后续需要真正切分 shared/tail 执行路径。
- `static_ptq_tail`：通过 vLLM shell 调用 HF/River/HQQ Python 模型，内部维护 HF KV cache，不使用 vLLM paged KV，只适合单请求原型验证。

## 速度结果

测试设置：prompt 约 64 tokens，输出 16 tokens，warmup 1，iters 3，`max_model_len=128`。

| mode | output TPS | mean latency |
|---|---:|---:|
| full_fp | 11.233 tok/s | 1.424 s |
| static_fp_tail | 11.236 tok/s | 1.424 s |
| static_ptq_tail | 1.104 tok/s | 14.493 s |

完整数据见 `summary.csv`。

## 现象与结论

- `full_fp` 和 `static_fp_tail` TPS 基本一致，符合当前实现预期，因为二者仍使用同一套 vLLM Llama 计算图。
- `static_ptq_tail` 能跑通，但 TPS 显著低于 FP；原因是它没有接入 vLLM 原生 kernel、paged KV、CUDA Graph，而是通过 Python/HQQ 逐层执行和动态 dequant。
- PTQ 初始化很慢：正式测试中模型加载约 443 s，峰值模型内存约 18.52 GiB；这不是生成 TPS，但说明当前 wrapper 不适合作为最终系统实现。
- 在 AGX 上默认 V1 多进程会在 NVML 初始化处 segfault；关闭 `VLLM_ENABLE_V1_MULTIPROCESSING` 后可稳定运行。

## 后续优化方向

- 将 PTQ tail 从 HF/HQQ wrapper 下沉到 vLLM 原生 model runner，避免 Python 层逐层调度。
- 为 HQQ/PTQ 权重实现 vLLM loader，避免 transformers 先忽略 HQQ metadata 再手动重读 shard。
- 实现真正 shared FP 层与 FP/PTQ tail 的静态路由；当前 FP tail 尚未获得结构性加速。
- 之后再接入 batch-aware/route-aware scheduler，避免不同 tail 请求混批时路径不一致。
