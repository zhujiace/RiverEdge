# 2026-09-07 RiverEdge 恢复与在线原型

2026-09-08 续跑已完成；优先阅读 `summary_20260908.md`。直答质量协议的局限、推理版真实子集结果、长输出重复测量及剩余风险均在那里记录。自动优化配置见 `autoresearch_proposal.md`，目前尚未启动。

本目录记录 `../../0907.md` 四步工作的实际结果。历史静态模型保持原状，新增模型通过插件名称 `RiverEdgeOnlineForCausalLM` 注册。没有修改 `third_party/vllm` 源码。

## 环境与基线

- Docker：`riveredge-vllm-src`，`vllm/vllm-openai:v0.24.0-aarch64`。
- vLLM / PyTorch / TorchAO：0.24.0 / 2.11.0+cu130 / 0.17.0。
- Checkpoint：容器内 `/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`。
- 同设备另有 SmolVLM 服务；未停止其他项目。数值是共享设备 smoke，不能用于严格历史复现或论文速度结论。
- `baseline.json/csv`：85 input、8 output、B=1、warmup=1、iters=2、FULL_DECODE_ONLY、Inductor 关闭。FP=10.9303、PTQ=21.0370 tok/s，8 个输出 token 一致，phase 正确。

基线命令：

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_vllm_unified_batch_cudagraph.py \
  --batch-sizes 1 --execution-modes cudagraph --max-new-tokens 8 --warmup 1 --iters 2 \
  --gpu-memory-utilization 0.36 \
  --output-json /workspace/vllm/RiverEdge/experiments/15_resume_20260907/baseline.json \
  --output-csv /workspace/vllm/RiverEdge/experiments/15_resume_20260907/baseline.csv
```

## 在线实现

- `models/online_llama.py`：单模型混合精度参考。按行分组 QKV/O/MLP，恢复原顺序后执行共享 attention，保留原 block table 和 slot mapping。
- `online_runtime.py`：从 runner 的实际请求顺序和已计算 token 数构造路由。支持 fp、ptq、mixed、batch hysteresis，以及 conservative 混合资格回退。
- `native`：保留原生 continuous batching，允许 FP prefill 和 PTQ decode 共存。
- `grouped`：调度前按 prefill/FP decode/PTQ decode 选组，复用原分配器，限制总 active requests，支持 EOS 补位、等待预算和 fairness。该实验适配器只在进程内安装，关闭后恢复原方法。
- `benchmark_riveredge_online.py`：真实引擎 step 重放，Poisson/bursty/JSONL 到达，逐 token delivery、TTFT/TPOT、SLO goodput、route trace 与电源轨采集。
- `head_ablation.py`：显式 `--head-variants int4` 才启用 HQQ LM-head 候选，只量化 PTQ decode 的输出投影；原 FP 权重及 checkpoint 不变。按 vocab blocks 量化以限制临时内存。

在线原型限制：同步调度、eager、TP=PP=1、无 speculative/prefix cache。KV 内存不足发生 preemption 时主动报错，因为尚未实现历史 route 重放。尚无动态 route CUDA Graph pool；禁止将 Python 路由 capture 成固定旧路径。资格来自 trace 的 `allow_ptq`，不是已实现 River confidence。

## 运行与分析

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/test_online_reference.py

docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_riveredge_online.py \
  --output /workspace/vllm/RiverEdge/experiments/15_resume_20260907/online_smoke.json \
  --requests 6 --batch-size 4 --max-tokens 8 --policies fp,ptq,mixed,batch \
  --schedulers native,grouped --low-batch 1 --high-batch 3 --force-length

python3 RiverEdge/river-vllm-edge/scripts/analyze_riveredge_online.py \
  RiverEdge/experiments/15_resume_20260907/online_smoke.json \
  --output RiverEdge/experiments/15_resume_20260907/online_analysis.json
```

第一轮 online smoke：8 配置 × 6 请求 × 8 token，20 项 phase/完成/同组检查通过；mixed/native 与 mixed/grouped 共 12 个请求与各自 FP/PTQ 基线逐 token 一致。FP/native=31.85 TPS，PTQ/native=47.59 TPS，mixed/native=25.40 TPS，mixed/grouped=24.98 TPS。当前结果不支持混合拆组加速；不能据此宣称完整系统优于 always-PTQ。

第一轮使用 1/3 人为 crossover 阈值，目的是小 smoke 内覆盖滞回切换，不是硬件性能阈值。第一次在线运行仍用自动 KV 预算；后续显式固定 256 MiB，避免共享 Orin 上内存 profiling 误配过大的 KV cache。

质量/LM-head/k 消融：

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_riveredge_online.py \
  --output /workspace/vllm/RiverEdge/experiments/15_resume_20260907/quality_head_ablation.json \
  --trace /workspace/vllm/RiverEdge/experiments/15_resume_20260907/quality_smoke.jsonl \
  --batch-size 4 --policies fp,ptq,mixed --schedulers native,grouped \
  --exit-layers 1,3 --head-variants fp,int4
```

`quality_smoke.jsonl` 为算术、知识、代码各两条手写检查，strict string match；只用于接线验证，不代表正式质量评估。其他数据可用相同 JSONL schema 导入。功耗字段为 VDD_GPU_SOC 电源轨，包括共享设备负载，不能称为独立整机能耗。首次 smoke 的功耗采样未覆盖完整首尾，记录了 coverage；后续版本补齐端点并将积分裁剪到请求计时窗口。

所有输出路径是显式参数，重跑时应使用新文件名保存已有结果。

## 修正后的质量结果

上面的 `quality_head_ablation.json` 使用裸文本，模型继续补写问答，strict string match 为零；该结果不用于质量结论。修正运行命令是：

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_riveredge_online.py \
  --output /workspace/vllm/RiverEdge/experiments/15_resume_20260907/quality_chat_ablation.json \
  --trace /workspace/vllm/RiverEdge/experiments/15_resume_20260907/quality_smoke.jsonl \
  --chat-template --batch-size 4 --policies fp,ptq,mixed --schedulers native \
  --exit-layers 1,3 --head-variants fp,int4
```

12 配置、72 次回答全部正确；24 项 phase/完成检查通过，功耗覆盖率均为 100%。每题只有两个输出 token（包含结束 token），不足以替代长 decode benchmark。

| k=3 模式 | FP head TPOT p50 | INT4 head TPOT p50 |
|---|---:|---:|
| PTQ | 56.59 ms | 54.18 ms |
| Mixed | 120.16 ms | 124.47 ms |

结论：INT4 head 仅保留为显式启用的候选，混合路径的索引/分块开销会抵消收益；默认 FP head、k=3 均不变。正式量化质量预算、长文本及长输出性能需要继续验证。

## 最终在线矩阵命令

```bash
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_riveredge_online.py \
  --output /workspace/vllm/RiverEdge/experiments/15_resume_20260907/online_final_matrix.json \
  --requests 6 --batch-size 4 --max-tokens 8 --policies fp,ptq,mixed,batch,conservative \
  --schedulers native,grouped --arrival-patterns poisson,bursty \
  --low-batch 1 --high-batch 3 --repeats 2 --force-length
```

`conservative` 对混合资格的 active batch 回退到 FP，避免同时执行双精度路径；它是保守实验策略，不是已训练的质量门控。所有输出通过 source SHA256 和版本记录追踪对应实现。
