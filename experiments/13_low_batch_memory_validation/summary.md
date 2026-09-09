# RiverEdge Docker 恢复、实验复现与内存带宽验证

日期：2026-07-22

## 1. 结论

1. **Docker 恢复成功。** 使用历史一致的 `vllm/vllm-openai:v0.24.0-aarch64`，RiverEdge 插件、统一 FP+HQQ checkpoint、FP/PTQ 两条路径均可正常推理。
2. **实验 11 复现成功。** Eager/CUDA Graph、FP/PTQ、batch=1/2/4/8/16 共 20 个点，TPS 相对历史结果的最大绝对偏差为 **1.07%**。
3. **单请求 FP decode 主要受权重流量和低算术强度限制，不是 HTTP/H2D 传输。** b1 时 GPU Active=99.71%，但 SM Issue=9.62%、活跃 warp=16.21%；EMC 稳态均值=79.66%、中位数=82%。GPU 一直工作，但不是计算单元饱和。
4. **PTQ 的低 batch 收益来自真实的权重流量压缩。** 同尺寸 `gate_up` kernel 的 L2 事务由 FP 的 238.65 MB 降至 PTQ 的 66.81 MB，减少 **3.57 倍**；CUDA Graph b1 TPS 从 11.27 提升到 24.55 tok/s（2.18 倍）。
5. **PTQ 的问题在大 batch kernel 扩展性，而非量化完全无效。** CUDA Graph 下 PTQ/FP 从 b1 的 2.18 倍降到 b8 的 1.30 倍，并在 b16 降到 0.946 倍。PTQ b4 的 EMC 反而低于 b1，说明此时瓶颈逐渐转向 INT4 kernel 的 batch 扩展、解包/反量化和执行效率。
6. 本实验中 FP b1→b2 为 1.96 倍、PTQ b1→b2 为 1.90 倍，**不能把“低并行度没有 TPS 收益”作为 vLLM 的普遍结论**。`profile/0706.md` 中小模型 server 数据需要按相同输入、输出、调度和计时口径单独复核。

## 2. 恢复环境

| 项目 | 配置 |
|---|---|
| 容器 | `riveredge-vllm-src` |
| 镜像 | `vllm/vllm-openai:v0.24.0-aarch64` |
| 镜像 digest | `sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b` |
| vLLM / PyTorch / TorchAO | 0.24.0 / 2.11.0+cu130 / 0.17.0 |
| 插件 | `/workspace/vllm/RiverEdge/river-vllm-edge`（editable install） |
| 模型 | `/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq` |
| 挂载 | `/home/orin/zjc/vllm:/workspace/vllm`、models、Hugging Face cache |
| 运行参数 | NVIDIA runtime、host network、host IPC、memlock=-1 |

`latest-aarch64` 已变化到 vLLM 0.25.1，不能用于历史复现。当前恢复容器保持运行：

```bash
sudo docker exec -it riveredge-vllm-src bash
```

`restore_smoke.json` 验证 FP/PTQ 输出 token 一致，PTQ phase trace 为 `prefill_fp,decode_ptq`。

## 3. TPS 复现

统一配置：BF16，k=3，85 input tokens，强制生成 32 tokens，warmup=1，测量 3 次，prefix cache/chunked prefill/Inductor 关闭，CUDA Graph 使用 `FULL_DECODE_ONLY`，AGX MAXN、EMC 3199 MHz。

| 执行 | Batch | FP 历史→复现 | PTQ 历史→复现 | PTQ/FP |
|---|---:|---:|---:|---:|
| Eager | 1 | 11.18→11.15 | 19.92→19.82 | 1.778 |
| Eager | 2 | 21.81→21.71 | 36.87→36.85 | 1.697 |
| Eager | 4 | 41.54→41.41 | 67.87→67.14 | 1.622 |
| Eager | 8 | 71.10→70.86 | 91.68→91.60 | 1.293 |
| Eager | 16 | 129.15→128.83 | 121.88→121.76 | 0.945 |
| CUDA Graph | 1 | 11.30→11.27 | 24.65→24.55 | 2.178 |
| CUDA Graph | 2 | 22.12→22.04 | 46.85→46.53 | 2.111 |
| CUDA Graph | 4 | 42.07→41.95 | 77.91→77.78 | 1.854 |
| CUDA Graph | 8 | 71.91→71.75 | 93.23→93.22 | 1.299 |
| CUDA Graph | 16 | 130.66→130.36 | 123.37→123.30 | 0.946 |

## 4. Nsight Systems 全图观察

以下为 CUDA Graph 稳态范围的 100 Hz GPU metrics 均值：

| 模式 | B | GPU Active | SM Active | SM Issue | Tensor Active | 活跃 warp | 未分配 warp |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | 1 | 99.71% | 97.62% | 9.62% | 26.30% | 16.21% | 81.37% |
| FP | 2 | 99.75% | 97.88% | 9.91% | 27.29% | 16.30% | 81.50% |
| FP | 4 | 99.72% | 98.01% | 9.90% | 27.28% | 16.45% | 81.39% |
| PTQ | 1 | 99.51% | 95.86% | 38.58% | 23.27% | 42.19% | 53.79% |
| PTQ | 2 | 99.51% | 96.41% | 37.68% | 25.31% | 41.24% | 55.12% |
| PTQ | 4 | 99.48% | 96.59% | 32.50% | 23.71% | 40.89% | 55.71% |

FP eager b1 中 BF16 GEMM 占 GPU kernel 时间 **97.8%**。PTQ eager b1 中 TorchAO INT4 `tinygemm` 占 42.0%，其余主要是 FP prefill、前三层 shared FP 和 LM head。profile 区间内两种模式的 H2D 都只有 **2,228 bytes**（单次最大 680 bytes），不存在逐 token 搬运模型权重的行为。

## 5. Kernel 级流量验证

NCU 使用 application replay，只选择一个代表性 b1 decode `gate_up` kernel：

| 指标 | FP BF16 GEMM | PTQ INT4 tinygemm |
|---|---:|---:|
| Grid / block | 448×1×1 / 128×1×1 | 1×3584×1 / 32×8×1 |
| L2 总事务 | 238.65 MB | 66.81 MB |
| NCU compute/memory throughput | 91.42% | 70.37% |
| SM Issue | 14.94% | 79.19% |
| 活跃 warp | 16.50% | 62.98% |
| 正常 nsys 平均 kernel 时间 | 1.322 ms | 0.447 ms |

L2 流量降低 3.57 倍，正常运行 kernel 时间降低约 2.96 倍。NCU replay 会显著放慢 kernel，因此报告中的 NCU duration 只用于同口径核对，不作为正常延迟。

Orin 的 `mcc__dram_throughput_*` 在 NCU 指标列表中存在，但本机对两个 kernel 均返回 `NaN`；`lts__d_sectors_fill_device` 也因统一内存语义返回 0。因此不能声称获得了精确的逐 kernel DRAM GB/s，物理内存压力由 tegrastats 交叉验证。

## 6. EMC 带宽验证

宿主机 `tegrastats` 以 100 ms 采样。选择最长连续 `GR3D>=90%` 区间，去除首尾各 1 秒；EMC 全程为 3199 MHz。NVIDIA 文档将 EMC 百分比定义为相对当前频率的已用内存带宽：[Tegrastats Utility](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)。

| 模式 | B | TPS | EMC 均值 | EMC 中位数 | P5–P95 | GR3D 均值 |
|---|---:|---:|---:|---:|---:|---:|
| FP | 1 | 11.13 | 79.66% | 82.0% | 63–83% | 99.12% |
| FP | 4 | 41.54 | 80.94% | 81.0% | 78–83% | 99.13% |
| PTQ | 1 | 24.34 | 74.58% | 75.0% | 73–76% | 99.00% |
| PTQ | 4 | 76.96 | 60.87% | 63.5% | 48–66% | 99.11% |

现象解释：

- FP b1 已有约 80% EMC 压力，但 SM Issue 很低，符合小 M GEMV/GEMM 的“高权重流量、低算术强度”特征。
- FP b1→b4 TPS 提高 3.73 倍，而 EMC 基本不变，说明同一次权重读取服务更多 token，权重成本被 batch 摊薄。
- PTQ b1 降低单步权重流量并缩短 TPOT；到 b4 后 EMC 降到约 61%，但 TPS 只增加 3.16 倍，说明量化 kernel 的计算/解包开销开始限制扩展。
- 因为 FP EMC 约为 80% 而非 100%，严谨表述应为“memory-traffic dominated / memory-pressure high”，不应写成“物理带宽完全饱和”。

## 7. 对 RiverEdge 的启示

1. 论文场景应定位为端侧低负载、活跃 batch 较小、TPOT/单请求吞吐敏感；PTQ early-exit 在 b1–b4 的收益有直接硬件证据。
2. 不要排斥 batching。更合理的运行策略是按 active batch 选择 tail：本机 k=3 下 b1–b8 选 PTQ，接近 b16 时切换 FP；阈值应随模型、输出长度和路由分布标定。
3. 下一优化重点是 batch-aware INT4 kernel：优化 M=2/4/8/16 tile、融合 scale/zero 解码、减少 D2D 临时操作，并为 FP/PTQ 请求分别组桶。
4. Naive route 应改成 route-aware continuous batching：调度器按 tail 类型和剩余长度形成稳定微批，避免每步拆批和串行执行双 tail。

## 8. Profiling 安全性

NCU 默认 kernel replay 会备份 18.5 GiB device allocation 到系统内存，实际触发 `NvMapMemAllocInternalTagged failed: error 12` 并将 swap 推高到约 1 GiB。这很可能是此前 VS Code 远程连接中断的原因。本次立即终止该方式，改用 application replay；后续 AGX 上禁止对完整模型使用默认 kernel replay。

## 9. 结果文件

- `reproduction_matrix.csv/json`：完整复现数据。
- `reproduction_comparison.csv`：历史与复现偏差。
- `reproduction_speedups.csv`：FP/PTQ speedup。
- `nsys_gpu_metrics.csv`、`nsys_memcpy_bytes.csv`：全图利用率与传输量。
- `ncu_kernel_metrics.csv`、`ncu/*.ncu-rep`：kernel 级流量和原始报告。
- `tegrastats_emc_metrics.csv`、`tegrastats/*.log`：EMC 汇总和原始采样。
- `validation_summary.json`：全部结构化结论。
