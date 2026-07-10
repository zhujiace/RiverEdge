# P4 PyTorch 双 Tail 原型与速度测试

## 实现

新增脚本：

- `RiverEdge/experiments/p4_pytorch_dual_tail_benchmark.py`

脚本直接加载当前 River checkpoint，不修改 River/vLLM 内部实现。手写执行三种路径：

- `full_fp`：`model.layers[0:32] + model.norm`
- `fp_tail`：`model.layers[0:k] + model.layers[k:32] + model.norm`
- `ptq_tail`：`model.layers[0:k] + model.exit_modules[0][k:32] + exit_norm`

输出文件：

- `RiverEdge/experiments/04_split_reference/split_forward_results_torchao.json`
- `RiverEdge/experiments/04_split_reference/tail_latency_torchao.csv`
- `RiverEdge/experiments/04_split_reference/split_forward_decode_batch_torchao.json`
- `RiverEdge/experiments/04_split_reference/tail_latency_decode_batch_torchao.csv`

## 实验设置

- Docker：`river-llama32-8b`
- 模型：`/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16`
- GPU：Orin
- dtype：bf16
- HQQ backend：`torchao_int4`
- warmup：2
- iters：5
- 计时：CUDA synchronize 后统计 median

## batch=1，不同 seq_len

表中数值为 `ptq_tail` 相对 `fp_tail` 的 median speedup。

| exit layer k | seq_len=1 | seq_len=16 | seq_len=128 |
|---:|---:|---:|---:|
| 1 | 1.451x | 0.982x | 0.200x |
| 3 | 1.450x | 0.986x | 0.211x |
| 8 | 1.446x | 0.993x | 0.244x |
| 16 | 1.452x | 1.005x | 0.326x |
| 24 | 1.197x | 1.010x | 0.491x |

现象：

- `seq_len=1`：PTQ tail 明显更快，`k<=16` 约 1.45x。
- `seq_len=16`：基本持平。
- `seq_len=128`：PTQ tail 明显更慢，不适合 prefill 长序列。

## decode 型 batch sweep

固定 `seq_len=1`，表中数值为 `ptq_tail` 相对 `fp_tail` 的 median speedup。

| batch | k=1 | k=3 | k=8 | k=16 | k=24 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.448x | 1.443x | 1.449x | 1.450x | 1.197x |
| 2 | 1.456x | 1.458x | 1.455x | 1.378x | 1.190x |
| 4 | 1.460x | 1.462x | 1.463x | 1.395x | 1.170x |

## 结论

P4 PyTorch 原型可运行，且当前 checkpoint 的 PTQ tail 对 decode 型小矩阵有实际加速；对 `k<=16` 的 batch 1/2/4，收益大多在 1.38x 到 1.46x。

但 PTQ tail 对长序列 prefill 很差，`seq_len=128` 只有 FP tail 的 0.20x 到 0.49x。因此后续系统设计应采用：

- prefill：优先保留 FP backbone/tail。
- decode：尝试 route 到 PTQ tail。
- scheduler：需要区分 prefill 与 decode，不应把长 prompt prefill 直接导向当前 PTQ tail。

当前测试是 no-cache PyTorch forward 原型，尚未验证完整 KV cache 写入和 vLLM scheduler 集成。

## Routed Reference 更新

新增文件：

- `RiverEdge/river-vllm-edge/river_vllm_ext/models/split_model_reference.py`
- `RiverEdge/river-vllm-edge/scripts/test_split_forward.py`

实现的路径：

- prefill：正常 full FP backbone，写入初始 KV cache。
- decode：先执行 shared FP prefix `layers[0:k]`。
- route：默认 `k=3`、`threshold=0.5`，用 layer k 输出与上一层 hidden state 的 cosine similarity 判定。
- PTQ tail：`exit_modules[0][k:32] + exit_norm`。
- FP tail：`layers[k:32] + model.norm`。

当前 reference 使用单个 mixed KV cache：prefill 写 FP KV；decode 阶段 prefix 写 FP KV，tail 按本步 route 写 FP 或 PTQ KV。该实现用于验证 PyTorch 路径与速度，不代表最终 vLLM paged-KV 设计已经完成。

测试命令：

```bash
sudo docker exec river-llama32-8b bash -lc 'cd /workspace/vllm && python3 RiverEdge/river-vllm-edge/scripts/test_split_forward.py --prompt-len 64 --max-new-tokens 16 --exit-layer 3 --route-threshold 0.5 --modes auto,ptq,fp --output-json /workspace/vllm/RiverEdge/experiments/04_split_reference/routed_reference_k3_thr05_modes.json --output-csv /workspace/vllm/RiverEdge/experiments/04_split_reference/routed_reference_k3_thr05_steps.csv'
```

结果：

| mode | decode steps | route counts | decode TPS | avg route score |
|---|---:|---|---:|---:|
| auto | 15 | PTQ 15 / FP 0 | 11.86 | 0.764 |
| force PTQ | 15 | PTQ 15 / FP 0 | 12.73 | 0.764 |
| force FP | 15 | PTQ 0 / FP 15 | 9.99 | 0.768 |

结论：

- 默认阈值 0.5 下，本次随机 prompt 的 decode step 全部路由到 PTQ。
- 带 KV cache 的 routed reference 中，force PTQ 相比 force FP 的 decode TPS 约为 `1.27x`。
- auto 低于 force PTQ，主要来自 route score 计算和动态分支开销。

## lm-evaluation 完整任务测试

新增脚本：

- `RiverEdge/river-vllm-edge/scripts/run_lm_eval_dataset_reference.py`

该脚本复用 `rivier/lm-evaluation` 的 `TaskManager/get_task_dict`，从真实 benchmark task 中调用 `doc_to_text` 构造 prompt，再用 routed reference 逐样本生成并统计速度、路由比例和输出一致性。

测试命令：

```bash
sudo docker exec river-llama32-8b bash -lc 'cd /workspace/vllm && python3 RiverEdge/river-vllm-edge/scripts/run_lm_eval_dataset_reference.py --tasks mmlu_abstract_algebra --limit 0 --max-new-tokens 8 --exit-layer 3 --route-threshold 0.5 --modes auto,ptq,fp --output-json /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_routed_k3_thr05_summary.json --output-csv /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_routed_k3_thr05_samples.csv'
```

数据集：

- task：`mmlu_abstract_algebra`
- 样本数：100
- prompt tokens：mean 63.44，median 62，min 32，max 110
- 每条生成：8 tokens，其中 prefill 后 decode 7 steps
- 总 decode steps：700

结果：

| mode | decode TPS | PTQ route ratio | route score mean | token match vs FP | exact seq match vs FP |
|---|---:|---:|---:|---:|---:|
| auto | 12.44 | 100% | 0.782 | 0.8925 | 0.71 |
| force PTQ | 12.47 | 100% | 0.782 | 0.8925 | 0.71 |
| force FP | 9.91 | 0% | 0.783 | 1.0000 | 1.00 |

结论：

- 在完整 `mmlu_abstract_algebra` 100 条样本上，默认阈值 0.5 仍然过低，auto 全部路由到 PTQ。
- PTQ decode TPS 相比 FP decode TPS 约 `1.26x`。
- PTQ 与 FP 的 greedy 输出并非完全一致：平均 token match 为 `89.25%`，整段 8-token 输出完全一致的样本比例为 `71%`。
- 后续需要做 threshold sweep 或质量约束路由，否则默认 0.5 更接近 `all_ptq` 策略，而不是混合 route。

## Threshold 0.8 重测

测试命令：

```bash
sudo docker exec river-llama32-8b bash -lc 'cd /workspace/vllm && python3 RiverEdge/river-vllm-edge/scripts/run_lm_eval_dataset_reference.py --tasks mmlu_abstract_algebra --limit 0 --max-new-tokens 8 --exit-layer 3 --route-threshold 0.8 --modes auto,ptq,fp --output-json /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_routed_k3_thr08_summary.json --output-csv /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_routed_k3_thr08_samples.csv'
```

结果：

| threshold | mode | decode TPS | PTQ route ratio | token match vs FP | exact seq match vs FP |
|---:|---|---:|---:|---:|---:|
| 0.5 | auto | 12.44 | 100.0% | 89.25% | 71% |
| 0.8 | auto | 10.51 | 22.43% | 96.88% | 89% |
| - | force PTQ | 13.31 | 100.0% | 89.25% | 71% |
| - | force FP | 9.92 | 0.0% | 100.00% | 100% |

结论：

- threshold `0.8` 能把 auto 从 all-PTQ 拉回混合路由，PTQ route ratio 为 `22.43%`。
- 输出一致性明显改善：exact sequence match 从 `71%` 提升到 `89%`。
- 速度收益变小：auto decode TPS `10.51`，相比 FP 的 `9.92` 约 `1.06x`。
- 当前 `0.8` 是质量更稳的默认值，但如果希望系统收益更明显，应继续 sweep `0.75-0.8` 区间。

## Batch Sweep

新增脚本：

- `RiverEdge/river-vllm-edge/scripts/run_lm_eval_dataset_batch_reference.py`

测试命令：

```bash
sudo docker exec river-llama32-8b bash -lc 'cd /workspace/vllm && python3 RiverEdge/river-vllm-edge/scripts/run_lm_eval_dataset_batch_reference.py --tasks mmlu_abstract_algebra --limit 0 --batch-sizes 1,2,4,8,16 --max-new-tokens 8 --exit-layer 3 --route-threshold 0.8 --modes auto,ptq,fp --output-json /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_batch_sweep_k3_thr08_summary.json --output-csv /workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_batch_sweep_k3_thr08_batches.csv'
```

速度结果。这里的 `decode token TPS` 是输出 token 吞吐，适合比较 batch scaling。

| batch | auto TPS | force PTQ TPS | force FP TPS | auto / FP | PTQ / FP |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.54 | 13.17 | 9.99 | 1.06x | 1.32x |
| 2 | 19.96 | 26.20 | 19.37 | 1.03x | 1.35x |
| 4 | 38.00 | 52.16 | 37.70 | 1.01x | 1.38x |
| 8 | 69.02 | 95.01 | 68.92 | 1.00x | 1.38x |
| 16 | 117.63 | 119.21 | 117.38 | 1.00x | 1.02x |

路由与一致性：

| batch | auto PTQ route ratio | auto token match vs FP | auto exact seq match |
|---:|---:|---:|---:|
| 1 | 22.43% | 96.88% | 89% |
| 2 | 11.14% | 99.00% | 95% |
| 4 | 2.86% | 99.88% | 99% |
| 8 | 0.57% | 100.00% | 100% |
| 16 | 0.57% | 100.00% | 100% |

现象：

- force PTQ/FP 都随 batch 增大而提高输出 token TPS，说明 batch 能有效摊薄 decode kernel launch 和 Python 调度开销。
- batch 1 到 8 时，force PTQ 相比 force FP 保持约 `1.32x-1.38x`；batch 16 时差距几乎消失，可能因为 batch 16 下 FP GEMM 利用率提升，PTQ 优势被压缩。
- 当前 auto 是 batch-level gate：只有整个 batch 都满足阈值才走 PTQ。因此 batch 越大，PTQ route ratio 越低；batch 8/16 基本退化为 FP。
- 如果要在 batch 下保留 PTQ 收益，下一步不能用整批 all-pass gate，需要做 per-sample route 后再把 PTQ/FP 样本拆成两个 microbatch。
