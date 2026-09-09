# RiverEdge vLLM 完整 GSM8K 真实负载实验

## 目的

补充 experiment 11 的固定长度实验：使用完整 GSM8K test split，让 Llama Instruct 按 EOS 自然停止，观察变长输入、变长输出和 batch drain 下的 FP/PTQ 性能与准确率。

## 配置

- 数据：`gsm8k/main/test`，完整 1319 条；采用 `gsm8k_cot_llama` 的 8-shot 多轮 chat template
- 权重：`Llama-3.1-8B-Instruct-riveredge-fp-hqq`，BF16，`k=3`
- 模式：`full_fp`；`static_ptq_tail`（prefill FP，decode HQQ layer 4-32）
- 调度：按原始顺序封闭 batch=16，共 83 组，最后一组 7 条
- 生成：greedy，`temperature=0`，不设置 `min_tokens`，不忽略 EOS
- `max_tokens=512` 仅防止异常循环；命中情况单独统计，不作为固定输出长度
- CUDA Graph：`FULL_DECODE_ONLY`，capture size 1/2/4/8/16；Inductor、prefix cache、chunked prefill 关闭
- `max_model_len=2048`，`max_num_seqs=16`，`gpu_memory_utilization=0.50`
- TPS 计时包含每组 FP prefill、自然长度 decode 和 vLLM 调度，不包含模型加载及 warmup

Prompt token 范围 1129-1292，均值 1165.70，p50 1162，p90 1196。Output token 不固定。

## 全量结果

| 指标 | Full FP | Static PTQ tail | PTQ 相对 FP |
|---|---:|---:|---:|
| 样本数 | 1319 | 1319 | - |
| 总输入 tokens | 1,537,553 | 1,537,553 | 0% |
| 总输出 tokens | 156,317 | 158,719 | +1.54% |
| 计时时间 | 3091.23 s | 2663.82 s | -13.83% |
| Output TPS | 50.57 | 59.58 | **+17.83%** |
| Requests/s | 0.4267 | 0.4952 | **+16.05%** |
| Strict accuracy | 82.71% | 82.41% | -0.30 pp |
| Flexible accuracy | 83.09% | 82.64% | -0.45 pp |
| EOS 停止 | 1313 (99.55%) | 1311 (99.39%) | -0.16 pp |
| 512-token 上限命中 | 6 (0.45%) | 8 (0.61%) | +2 条 |

输出长度：FP 均值/p50/p90 为 118.51/106/178.2；PTQ 为 120.33/107/185。PTQ 并未通过大幅增加输出数虚增 TPS，总输出仅增加 2402 tokens。

## 分块与配对结果

| 指标 | Full FP | Static PTQ tail |
|---|---:|---:|
| Chunk latency p50 / p90 / max | 35.19 / 48.93 / 61.20 s | 31.44 / 39.04 / 43.87 s |
| Chunk output TPS p50 / p90 / max | 53.62 / 58.33 / 62.46 | 60.14 / 64.57 / 67.93 |

Strict accuracy 的逐样本配对如下：

| 两者都对 | 仅 FP 对 | 仅 PTQ 对 | 两者都错 | McNemar exact p |
|---:|---:|---:|---:|---:|
| 1019 | 72 | 68 | 160 | 0.800 |

准确率净差仅 4 条，未达到统计显著。两模式 strict prediction 一致 1093/1319（82.87%），生成文本完全一致 135/1319（10.24%），说明量化明显改变推理过程，但最终答案大多保持一致。PTQ 输出更短/相同/更长的样本分别为 496/227/596。

## 现象与结论

1. 在完整真实 workload 下，PTQ tail 将 output TPS 提升 17.83%，同时将完成整个数据集的时间缩短 13.83%；strict accuracy 仅下降 0.30 pp，差异不显著。
2. 该结果不与固定长度 batch=16 时 PTQ 慢于 FP 的结果冲突。变长请求会陆续 EOS，活跃 decode batch 很快从 16 降到 8/4/2/1；experiment 11 已证明 PTQ 在这些小 batch 区间更快，因此真实任务重新获得 PTQ 优势。
3. 真实 TPS 仍显著低于固定 32-token 的满 batch TPS。主要原因是 8-shot 长 prefill 被计时，以及封闭 batch 中短请求结束后槽位不补充，剩余长请求造成 batch drain。
4. 本实验不是 online continuous batching。它刻意使用封闭分块，以保证 PTQ 模式中 prefill 全 FP、纯 decode 才进入 PTQ tail；当前实现若把新 prefill 与 decode 混在同一 forward，整批会回退到 FP。
5. 下一阶段最有价值的优化是将 FP prefill 与 FP/PTQ decode 分离调度，在 decode 槽位空出时补充同一路径请求；其次可做输出长度分桶，降低静态实验中的长尾空洞。

## 复现命令

```bash
sudo docker exec riveredge-vllm-src python3 \
  /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_vllm_gsm8k.py
```

## 文件

- 汇总：`gsm8k_full_cudagraph.json`、`gsm8k_full_cudagraph.csv`
- 模式汇总：`gsm8k_full_cudagraph.{full_fp,static_ptq_tail}.summary.json`
- 逐样本：`gsm8k_full_cudagraph.{full_fp,static_ptq_tail}.samples.jsonl`
- 分块：`gsm8k_full_cudagraph.{full_fp,static_ptq_tail}.chunks.csv`
- 配对：`gsm8k_full_cudagraph.paired.json`
- 数据：`gsm8k_test.jsonl`
- 脚本：`river-vllm-edge/scripts/benchmark_vllm_gsm8k.py`
