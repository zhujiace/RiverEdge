# P5 构建与运行记录

## 已完成

1. 在 AGX 上拉取 vLLM `v0.24.0` 源码，路径为 `RiverEdge/river-vllm-edge/third_party/vllm`。
2. 创建源码开发分支 `riveredge-dev`。
3. 启动源码开发容器 `riveredge-vllm-src`。
4. 安装容器内缺失依赖：`git`、`cmake`、`setuptools-rust`、`wheel`、`build`。
5. 完成 `pip install -e . --no-build-isolation` editable 安装。
6. 完成源码版 vLLM import、扩展加载、server、completion smoke test。

## 日志文件

- 首次失败日志：`build_log.txt`
- 成功安装日志：`build_precompiled_retry.log`
- 默认 server 启动日志：`server_smoke.log`
- eager server 启动日志：`server_smoke_eager.log`
- 纯文本 eager server 启动日志：`server_smoke_lm_only_eager.log`
- completion 响应：`completion_smoke_response.json`

## 关键现象

- 第一次 editable 安装失败，原因是容器内缺 `git`，`setuptools-scm` 无法识别源码版本。
- 加入 `git safe.directory` 并设置 `VLLM_VERSION_OVERRIDE=0.24.0` 后安装成功。
- 默认 server 会进入 torch.compile/CUDA Graph 初始化，AGX 上启动较慢。
- Qwen3.5-0.8B 带 `vision_config`，默认会进入多模态 encoder cache profiling；加 `--language-model-only` 后可快速完成纯文本 server 验证。
- curl 默认使用宿主机 `ALL_PROXY`，请求本地 vLLM server 时需要 `--noproxy '*'`。

## 当前限制

当前基线没有重新从源码编译 C++/CUDA/Rust 扩展，而是使用与官方镜像 ABI 匹配的预编译扩展。后续修改 Python 层可直接生效；修改 native extension 时需要补做完整本地编译。
