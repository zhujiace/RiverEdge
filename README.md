# River-style Early-Exit LLM Serving on Edge vLLM

当前实现的统一优化开关与原生 vLLM 对照入口见
[river-vllm-edge/README.md](river-vllm-edge/README.md)。新入口默认关闭全部
RiverEdge 优化，使用原生模型类和 FP checkpoint；历史 profiling 脚本仍保留原模式参数。

## 0. 项目定位

本项目的核心目标不是重新提出一种新的提前退出算法，而是将 **River-style Transformer early-exit 模型**高效部署到端侧 vLLM serving 框架中，并围绕 **TPS / TPOT** 进行系统优化。

更准确地说，本项目关注的是：

> 如何将 River-style variable-depth early exit 转换为适合 vLLM continuous batching、CUDA Graph 复用和 paged KV cache 管理的 fixed-checkpoint routed serving runtime。

在当前阶段，我们暂不考虑 prefill 优化，重点聚焦 decode 阶段，因为 TPOT 和输出 TPS 主要由 decode 阶段决定。

---

## 1. 问题背景

### 1.1 自回归 LLM decode 的系统瓶颈

LLM 自回归推理在 decode 阶段每次生成一个新 token。每个 decode step 都需要：

```text
读取历史 KV cache
执行 Transformer layers
写入当前 token 的新 KV
采样下一个 token
```

在端侧 GPU 上，decode 阶段面临几个主要瓶颈：

```text
1. batch size 通常较小，GPU 利用率不稳定
2. 每个 token 都需要完整 forward，TPOT 较高
3. KV cache 不断增长，占用显存/统一内存
4. 小 batch 下 CUDA launch overhead 更明显
5. 功耗、带宽、热限制会影响持续 TPS
```

vLLM 通过 continuous batching、PagedAttention 和 paged KV cache 提升 serving throughput，但原生 vLLM 默认所有请求执行同一套完整模型路径，并没有直接支持 River-style variable-depth early exit。

---

### 1.2 River-LLM 解决了算法层面的 early-exit KV 问题

River-LLM 的目标是让 decoder-only LLM 能够实现 seamless early exit。它在一定程度上解决了传统 decoder-only early exit 中的 KV Cache Absence 问题，即：

```text
如果某个 token 在中间层提前退出，
那么后续被跳过的 deeper layers 没有生成对应的 KV。
```

River 通过 KV share 等机制缓解这一问题，使 early-exit token 后续仍然可以参与自回归生成。

因此，本项目不把以下问题作为核心研究对象：

```text
1. early-exit token 的生成质量
2. PTQ / FP 产生 KV 的误差分析
3. missing KV 的语义修复
4. River 本身的 early-exit 判定算法
```

我们的研究重点是：

> River-style early-exit 模型在端侧 vLLM serving 中如何真正转化为 wall-clock TPS / TPOT 提升。

---

## 2. 关键系统问题

### 2.1 Naive batch execution 容易被最深退出 token 限制

River-style early exit 允许不同 token 在不同层退出。例如，一个 batch 中可能出现：

```text
token A exits at layer 3
token B exits at layer 5
token C exits at layer 12
token D exits at layer 20
```

如果 runtime 采用 naive synchronous layer-by-layer batching，那么整个 batch 的实际执行时间容易受到最深退出 token 的限制。

也就是说，浅层退出 token 虽然理论上可以很早完成，但在同步 batch 中可能需要等待仍在深层执行的 token。这样会导致：

```text
1. 浅层退出 token 的 latency benefit 无法立即体现
2. 当前 batch iteration 的 wall-clock time 接近最晚退出 token 的执行时间
3. early exit 的理论 layer saving 无法完全转化为 TPOT 降低
```

因此，River-style early exit 要在 serving 中获得真实加速，不能只依赖算法层面的平均退出层数降低，还必须解决 **depth divergence 下的 batch execution 问题**。

更准确地说：

> Naive synchronous variable-depth batching suffers from deepest-active-token limitation unless the runtime introduces route-aware micro-batching, compaction, or rescheduling.

---

### 2.2 Variable exit depth 使 batch 组织和 CUDA Graph 复用困难

River 允许 token 在任意层退出：

```text
exit layer ∈ {1, 2, ..., L}
```

在 serving runtime 中，这会产生大量不同的 execution paths：

```text
exit@1
exit@2
exit@3
...
exit@L
```

这会带来两个直接问题。

首先，batch 很难稳定组织。不同 token 或 request 的有效执行深度不同，导致 batch 内部出现 depth divergence。即使 runtime 按退出层进行 bucket，也会造成：

```text
1. bucket 数量较多
2. 每个 bucket 的 batch size 变小
3. 调度和 compaction 开销增加
4. 端侧 GPU 上小 batch 问题更加明显
```

其次，CUDA Graph 复用困难。CUDA Graph 通常要求较稳定的执行路径和 shape。variable-depth early exit 会产生许多 graph families 或 capture variants：

```text
Graph[exit@3][batch_size]
Graph[exit@5][batch_size]
Graph[exit@12][batch_size]
...
```

在端侧 GPU 上，这会导致：

```text
1. CUDA Graph capture / replay 命中率下降
2. graph 管理复杂度上升
3. fallback eager 的比例增加
4. 小 batch 下 launch overhead 更明显
```

因此，原始 River-style 任意退出层虽然算法上灵活，但并不天然适合 vLLM-style continuous batching 和端侧 CUDA Graph 优化。

---

### 2.3 KV cache 需要与 vLLM paged KV cache 兼容

在本项目中，每个 token 在后半段只会选择一种 continuation route：

```text
shared FP layers
  ↓
route decision
  ↓
PTQ tail 或 FP tail
```

也就是说，对于后续 layers：

```text
如果 token 走 PTQ tail:
  KV 由 PTQ layers 产生

如果 token 走 FP tail:
  KV 由 FP layers 产生
```

但无论 KV 来自 PTQ tail 还是 FP tail，都应该写入同一个 token position 对应的 vLLM paged KV slot。

因此，本项目不应重写 vLLM 的 KV cache 系统，而应复用 vLLM 的：

```text
1. block table
2. slot mapping
3. paged KV block allocation
4. logical layer id
5. attention metadata
```

需要扩展的是执行语义：

```text
1. shared layers 写入前几层 KV
2. selected tail route 写入后续 layers KV
3. 一个 token 的 tail KV 只由 PTQ 或 FP route 写入一次
4. route 影响执行路径，不影响 KV slot allocation
```

更准确地说：

> 本项目需要实现 paged-KV-compatible routed execution，而不是重新设计 KV cache。

---

## 3. 研究动机

### 3.1 从 variable-depth early exit 到 fixed-checkpoint routed continuation

本项目的核心观察是：

> River-style exit depth 在某些模型和任务上可能高度集中在早期层附近。

例如，对于 32 层 Llama3.2-8B，如果观察到 River 的平均退出层集中在第 3 层附近，那么可以将原始任意退出层设计简化为固定检查点：

```text
Original River:
  exit layer ∈ {1, 2, ..., 32}

Our transformation:
  fixed checkpoint k = 3
  route ∈ {PTQ-tail, FP-tail}
```

模型因此被拆成三部分：

```text
Part A:
  layers 1–3 FP shared backbone

Part B:
  layers 4–32 PTQ tail

Part C:
  layers 4–32 FP tail
```

这个转换的意义是：

> 用固定 early checkpoint 牺牲一部分 variable-depth 灵活性，换取 serving runtime 的结构稳定性。

---

### 3.2 固定检查点降低系统复杂度

固定检查点后，原本大量 execution paths 被压缩为三个 graph families：

```text
Graph family 1:
  shared FP layers 1–3

Graph family 2:
  PTQ tail layers 4–32

Graph family 3:
  FP tail layers 4–32
```

注意，这不意味着系统中只有三个 CUDA Graph 实例。由于 CUDA Graph 仍然依赖 batch size / capture size，实际会是：

```text
Graph[shared][batch_size bucket]
Graph[ptq_tail][batch_size bucket]
Graph[fp_tail][batch_size bucket]
```

但是相比任意退出层带来的大量 variable-depth graph families，这种结构显著更适合端侧部署。

因此，本项目的动机可以概括为：

> 将 River-style variable-depth early exit 转换为 fixed-checkpoint routed continuation，从而使 early exit 更适合 vLLM continuous batching、CUDA Graph 复用和 paged KV cache 管理。

---

## 4. 方法实现

### 4.1 模型结构

以 32 层模型为例，设置固定检查点 `k = 3`：

```text
layers 1–3:
  FP shared backbone

layers 4–32:
  PTQ tail

layers 4–32:
  FP tail
```

decode token 的执行流程为：

```text
current token
  ↓
layers 1–3 FP shared backbone
  ↓
route decision
  ↓
┌─────────────────────┬─────────────────────┐
│ PTQ tail 4–32       │ FP tail 4–32         │
└─────────────────────┴─────────────────────┘
  ↓
lm_head / sampler
  ↓
append new token
```

其中 route decision 可以来自 River 的 early-exit criterion，也可以在项目初期使用简化的 gate 逻辑实现。

---

### 4.2 Decode 队列划分

原生 vLLM 中主要区分 prefill 和 decode。当前项目暂不考虑 prefill 优化，因此重点将 decode 拆成三个子阶段：

```text
Q_pre_decode
Q_ptq_decode
Q_fp_decode
```

#### Q_pre_decode

所有 active decode requests 都先进入该队列，执行：

```text
current token
  ↓
layers 1–3 FP
  ↓
route decision
```

这是所有 decode token 的入口，因此优先级最高。

#### Q_ptq_decode

走 PTQ tail 的 token/request 进入该队列，执行：

```text
layers 4–32 PTQ
```

该队列是快路径，目标是提升平均 TPS 并降低 easy token 的 TPOT。

#### Q_fp_decode

走 FP tail 的 token/request 进入该队列，执行：

```text
layers 4–32 FP
```

该队列是慢路径，用于处理 hard token 或 fallback token。它不能长期饥饿，否则会导致 P95/P99 TPOT 恶化。

---

### 4.3 Route-aware micro-batching

micro-batch 是本项目必须实现的关键机制。

一个 decode step 不再将所有 request 混在一起同步执行完整模型，而是：

```text
Step 1:
  从 Q_pre_decode 取出一批 token
  组成 shared micro-batch
  执行 layers 1–3 FP

Step 2:
  根据 route decision 分流

Step 3:
  PTQ route token 进入 Q_ptq_decode
  形成 PTQ micro-batch
  执行 layers 4–32 PTQ

Step 4:
  FP route token 进入 Q_fp_decode
  形成 FP micro-batch
  执行 layers 4–32 FP

Step 5:
  各自完成 sampler
  进入下一轮 Q_pre_decode
```

理想情况下，PTQ micro-batch 和 FP micro-batch 不应被强制绑定在同一个同步 iteration 中。

如果 PTQ micro-batch 先完成，它应该可以先进行 sampling，并进入下一轮 `Q_pre_decode`，而不是等待 FP micro-batch 完成。

因此，本项目的 runtime 更接近：

```text
semi-asynchronous route-aware microbatch pipeline
```

而不是：

```text
synchronous routed batch
```

---

### 4.4 Scheduler 优先级

初始 scheduler 可以采用如下优先级：

```text
Priority 1: Q_pre_decode
Priority 2: Q_ptq_decode
Priority 3: Q_fp_decode
```

其理由是：

```text
Q_pre_decode:
  所有 decode token 的入口，决定后续 route

Q_ptq_decode:
  快路径，优先调度可以提升平均 TPS / TPOT

Q_fp_decode:
  慢路径，但需要 fairness，避免 hard token 长期等待
```

更稳妥的策略应加入 fairness：

```text
每执行 N 个 PTQ micro-batch，至少执行 1 个 FP micro-batch
```

或者基于 queue pressure 的动态调度：

```text
score(queue) =
  α · waiting_time
+ β · expected_TPS_gain
- γ · estimated_runtime
+ δ · deadline_pressure
```

第一阶段可以先实现简单规则：

```text
1. Q_pre_decode 最高优先级
2. Q_ptq_decode 优先于 Q_fp_decode
3. Q_fp_decode 使用周期性 fairness 机制避免饥饿
```

后续再实现 weighted fair scheduler 并做 ablation。

---

### 4.5 CUDA Graph 复用

固定检查点后，可以为三类 graph family 分别维护 CUDA Graph pool：

```text
GraphPool[shared][batch_size_bucket]
GraphPool[ptq_tail][batch_size_bucket]
GraphPool[fp_tail][batch_size_bucket]
```

例如：

```text
Graph[shared][1, 2, 4, 8, 16]
Graph[ptq_tail][1, 2, 4, 8, 16]
Graph[fp_tail][1, 2, 4, 8]
```

这样可以将原本 variable-depth early exit 中的大量 CUDA Graph 路径压缩成少数可复用 graph families。

该设计的目标是：

```text
1. 提高 CUDA Graph replay 命中率
2. 降低 fallback eager 比例
3. 提高端侧小 batch 下的执行稳定性
4. 减少 runtime graph 管理复杂度
```

---

### 4.6 Paged KV cache 兼容执行

对每个 decode token，KV 写入流程应为：

```text
1. vLLM KV manager 为当前 token position 分配 slot_mapping
2. shared layers 1–3 使用该 slot_mapping 写入 layer 1–3 KV
3. route decision 得到 PTQ 或 FP
4. selected tail route 使用同一个 slot_mapping 写入 layer 4–32 KV
5. sampler 输出下一个 token
```

需要满足以下约束：

```text
1. 一个 token 的 tail KV 只写一次
2. PTQ tail 和 FP tail 使用兼容的 KV layout
3. route 不改变 block table
4. route 不改变 logical layer id
5. route metadata 只决定执行哪个 tail module
```

第一阶段建议让 PTQ tail 和 FP tail 写入统一 KV dtype，避免 route-specific KV dtype 带来的 attention kernel 和 paged KV 管理复杂度。

---

## 5. 主要贡献

### Contribution 1: Fixed-checkpoint transformation for River-style early exit

本项目将 River-style variable-depth early exit 转换为 fixed-checkpoint routed continuation：

```text
variable exit depth:
  exit layer ∈ {1, 2, ..., L}

fixed checkpoint:
  shared layers 1–k
  route ∈ {PTQ-tail, FP-tail}
```

该转换减少 execution path 数量，提高 CUDA Graph 复用率，并使 River-style early exit 更适合 vLLM-style serving。

---

### Contribution 2: Route-aware micro-batching for decode

本项目将 decode 拆分为：

```text
pre-decode
PTQ-decode
FP-decode
```

并为 PTQ route 和 FP route 分别组织 micro-batch，从而缓解 depth divergence 和 deepest-token limitation。

---

### Contribution 3: Paged-KV-compatible routed execution

本项目复用 vLLM 的 paged KV cache，不重写 KV manager，而是扩展执行路径：

```text
shared layers 写入前几层 KV
selected tail route 写入后续层 KV
```

从而使 routed PTQ/FP continuation 与 vLLM block table、slot mapping 和 PagedAttention 兼容。

---

### Contribution 4: TPS/TPOT-oriented route-aware scheduler

本项目设计面向 decode TPS / TPOT 的 scheduler：

```text
Q_pre_decode > Q_ptq_decode > Q_fp_decode
```

并引入 fairness 机制，避免 FP route 饥饿。该 scheduler 的目标不是单纯最大化 batch size，而是在端侧设备上优化：

```text
1. TPS
2. TPOT / ITL
3. P95 / P99 TPOT
4. CUDA Graph hit rate
5. micro-batch utilization
6. KV block usage
7. energy per token
```

---

## 6. 项目可行性

### 6.1 算法前提：exit layer 分布集中

固定第 3 层作为 checkpoint 必须由实验支撑。需要统计：

```text
1. exit layer histogram
2. exit layer CDF
3. mean / median / P90 exit layer
4. 不同任务上的 exit 分布
5. k = 1, 2, 3, 4, 6, 8 的 sensitivity
```

如果大多数 token 的退出层确实集中在第 3 层附近，那么 fixed-checkpoint transformation 是合理的。

---

### 6.2 系统前提：PTQ tail 在端侧有实际收益

PTQ tail 是否能提升速度需要在目标端侧平台上实测。尤其在 AGX Orin 等设备上，PTQ 不一定天然快于 FP16，因为可能存在：

```text
1. INT8 / INT4 kernel backend 不充分优化
2. dequantization overhead 抵消收益
3. 小 batch 下 CUDA launch overhead 占主导
4. memory bandwidth 成为瓶颈
```

因此，需要测试：

```text
1. FP tail latency
2. PTQ tail latency
3. 不同 batch size 下 tail latency
4. 不同 quantization backend 的速度
5. PTQ route ratio 对整体 TPS / TPOT 的影响
```

---

### 6.3 工程前提：micro-batch 必须真实异步或半异步

如果系统只是同步执行：

```text
shared
  ↓
PTQ tail
  ↓
FP tail
  ↓
统一 sampling
```

那么 PTQ route 仍然会被 FP route 拖慢。

因此，要真正优化 TPOT，需要至少实现半异步 micro-batch pipeline：

```text
PTQ micro-batch 完成后先 sampling，并进入下一轮 Q_pre_decode
FP micro-batch 完成后再 sampling，并进入下一轮 Q_pre_decode
```

这也是本项目和 naive routed execution 的关键区别。

---

## 7. 实验设计建议

### 7.1 Baselines

建议至少比较：

```text
1. 原生 vLLM full FP
2. 原生 vLLM full PTQ
3. Static split: layers 1–3 FP + layers 4–32 PTQ
4. Naive routed execution without micro-batching
5. Route-aware micro-batching without CUDA Graph pool
6. Full system: route-aware micro-batching + graph pool + priority scheduler
```

---

### 7.2 Metrics

核心指标：

```text
TPS
TPOT / ITL
P50 / P95 / P99 TPOT
CUDA Graph hit rate
micro-batch size distribution
PTQ route ratio
FP route ratio
scheduler overhead
KV block usage
GPU utilization
power / energy per token
```

质量指标可以作为 sanity check，而不是核心优化目标：

```text
accuracy / pass@1 / perplexity / answer match
```

---

### 7.3 Ablation

关键 ablation：

```text
1. checkpoint k: 1, 2, 3, 4, 6, 8
2. scheduler priority policy
3. PTQ-first vs fairness scheduling
4. with / without CUDA Graph pool
5. synchronous routed batch vs semi-asynchronous micro-batch
6. different PTQ backend
```

---

## 8. 最终项目总结

本项目可以概括为：

> 将 River-style variable-depth early exit 转换为 fixed-checkpoint routed continuation，并在 vLLM 上实现 route-aware micro-batching、route-specific CUDA Graph reuse、paged-KV-compatible execution 和 TPS/TPOT-oriented scheduler，从而让 early exit 在端侧 serving 中真正转化为 wall-clock speedup。

核心技术路线是：

```text
River-style early exit
  ↓
fixed checkpoint k = 3
  ↓
shared FP layers 1–3
  ↓
route decision
  ↓
PTQ tail / FP tail
  ↓
route-aware micro-batching
  ↓
paged KV compatible execution
  ↓
scheduler optimized for TPS / TPOT
```

该项目的主要价值不在于提出新的 early-exit 判定方法，而在于：

> 解决 River-style early-exit Transformer 在端侧 vLLM serving 中的 batch divergence、CUDA Graph fragmentation 和 paged KV integration 问题。
