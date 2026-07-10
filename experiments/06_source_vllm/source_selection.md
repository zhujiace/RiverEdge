# P5 vLLM 源码基线选择

## 目标

在当前 AGX 上建立可修改、可运行的 vLLM 源码环境，作为后续 RiverEdge 接入 vLLM 的基础。

## 版本选择

- 官方运行镜像：`vllm/vllm-openai:latest-aarch64`
- 镜像内 vLLM：`0.24.0`
- Python：`3.12.13`
- PyTorch：`2.11.0+cu130`
- CUDA：`13.0`
- GPU：`Orin`, compute capability `8.7`

源码拉取位置：

```bash
RiverEdge/river-vllm-edge/third_party/vllm
```

源码版本：

```bash
git clone --depth 1 --branch v0.24.0 https://github.com/vllm-project/vllm.git RiverEdge/river-vllm-edge/third_party/vllm
cd RiverEdge/river-vllm-edge/third_party/vllm
git checkout -b riveredge-dev
```

- 分支：`riveredge-dev`
- tag：`v0.24.0`
- commit：`ee0da84ab9e04ac7610e28580af62c365e898389`

## 容器

开发容器名：`riveredge-vllm-src`

挂载：

- `/home/orin/zjc/vllm:/workspace/vllm`
- `/media/orin/Data/models:/models`
- `/media/orin/Data/huggingface:/disk/dataset/huggingface`

进入容器：

```bash
docker exec -it riveredge-vllm-src bash
```

## 安装方式

当前基线采用 editable 安装，并复用官方 aarch64/cu130 预编译扩展：

```bash
cd /workspace/vllm/RiverEdge/river-vllm-edge/third_party/vllm
VLLM_VERSION_OVERRIDE=0.24.0 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
python3 -m pip install -e . --no-build-isolation
```

该方式下 Python 代码来自源码树，适合修改调度、模型执行、entrypoint 等 Python 层逻辑。若后续修改 C++/CUDA/Rust 扩展，需要切换到 `VLLM_USE_PRECOMPILED=0` 并做完整本地编译。

## 验证

import 验证：

```text
vllm_file /workspace/vllm/RiverEdge/river-vllm-edge/third_party/vllm/vllm/__init__.py
vllm_version 0.24.0
```

server smoke test 使用 Qwen3.5-0.8B。该模型带 `vision_config`，默认会触发多模态 encoder cache profiling；纯文本验证需加 `--language-model-only`。

```bash
vllm serve /models/Qwen3.5-0.8B \
  --served-model-name qwen3.5-0.8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.50 \
  --max-num-seqs 4 \
  --enforce-eager \
  --language-model-only
```

验证结果：

- `/health` 返回成功。
- `/v1/models` 返回 `qwen3.5-0.8b`。
- `/v1/completions` 成功生成 8 个 token。

注意：宿主机存在 `ALL_PROXY=socks5://127.0.0.1:7890`，本地 curl 需要加 `--noproxy '*'`。
