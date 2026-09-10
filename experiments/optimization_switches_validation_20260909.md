# 优化开关验证（2026-09-09）

## 实现边界

统一入口：`river-vllm-edge/scripts/benchmark_vllm_switches.py`；复用接口：
`river_vllm_ext.optimization_switches.create_engine`。全部 RiverEdge 开关默认关闭，
选用原生 FP checkpoint 与原生 `LlamaForCausalLM`，不安装路由/调度/head 钩子。
此处不关闭 vLLM 自带的优化，也不改写 checkpoint。历史 profiling 脚本保留原参数，
新的 all-off 对照使用统一入口。开关组合和限制见 `river-vllm-edge/README.md`。

本次未修改 vendored vLLM 源码，其工作树干净，上游提交为 `ee0da84`。
未停止同机其他服务，未运行 INT4-head 质量实验。

## GPU smoke 结果

环境：Jetson AGX Orin；`riveredge-vllm-src`；vLLM 0.24.0、Torch 2.11.0+cu130、
TorchAO 0.17.0。每种配置 4 请求，B 上限 2，强制输出 8 token，预热 1 次，
测量 1 次，max_model_len=128，显式 KV 预算 128 MiB。k=3，使用 FP head。

| 路径 | 实际模型类/包装器 | 在线钩子 | 完成情况 |
| --- | --- | --- | --- |
| 全部关闭，显式 native eager | `vllm.model_executor.models.llama.LlamaForCausalLM` | 无 | 4 请求、32 token |
| PTQ tail + decode Graph | `vllm.compilation.cuda_graph.CUDAGraphWrapper` | 无 | 4 请求、32 token |
| PTQ tail + online + grouped + batch-aware | `river_vllm_ext.models.online_llama.RiverEdgeOnlineForCausalLM` | 有 | 4 请求、32 token |

- 原生日志确认 `quantization=None`，保留原生 prefix caching、chunked prefill 和
  异步调度。为了限定 smoke 成本，这次原生 GPU 运行显式使用 eager；未另跑原生
  默认 Inductor/Graph。CPU mock 测试确认全部关闭时不注入执行模式或模型覆盖参数。
- 静态路径成功捕获 B=1、B=2 的 full-decode Graph，图池约 0.03 GiB。
- 在线路径 31 个步骤，prefill 全为 FP，每一步 phase/route 同组；decode 中
  同时观察到 FP 与 PTQ，非只有配置标签变化。
- 原生路径验证之后新增了非布尔开关拒绝检查；其默认布尔配置与原生执行路径未改变，
  并通过后续 CPU 回归。各原始输出保留当时记录的源文件哈希。
- 这些结果只证明所列配置的功能可运行，不证明速度提升、一般数值等价或质量。
  INT4 head 的安装/恢复及保守回退通过 CPU 测试，本轮未做对应 GPU 消融。

## 实际命令

以下命令在宿主机仓库根目录执行；复跑时必须换用新的输出目录。

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_vllm_switches.py --all-off --enforce-eager --output /workspace/vllm/RiverEdge/reports/switch_smoke_XA67uTSc/native.json
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_vllm_switches.py --ptq-tail --decode-cudagraph --output /workspace/vllm/RiverEdge/reports/switch_smoke_XA67uTSc/static_graph.json
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_vllm_switches.py --ptq-tail --online-routing --grouped-scheduling --batch-aware-routing --output /workspace/vllm/RiverEdge/reports/switch_smoke_XA67uTSc/online_grouped.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=river-vllm-edge python3 -m unittest discover -s river-vllm-edge/scripts -p 'test_*switch*.py'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=river-vllm-edge python3 -m unittest discover -s river-vllm-edge/scripts -p 'test_online*.py'
```

32 项 CPU 测试通过：14 项开关测试、18 项现有在线/分析器测试。
新增 Python 模块语法检查及 `git diff --check` 通过。

本地原始 JSON 位于 `reports/switch_smoke_XA67uTSc/`，按当前 `.gitignore` 策略
不自动纳入 Git。原生权重目录为 `/models/Llama-3.1-8B-Instruct-native-view`，
双尾权重目录为 `/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`。
