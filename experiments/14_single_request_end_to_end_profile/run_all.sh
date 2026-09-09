#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-riveredge-vllm-src}"
CONTAINER_ROOT="/workspace/vllm/RiverEdge/experiments/14_single_request_end_to_end_profile"
NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
NSYS_HOME="${NSYS_HOME:-/opt/nvidia/nsight-systems/${NSYS_VERSION}}"
HOST_NSYS="${HOST_NSYS:-/usr/local/cuda-13.2/bin/nsys}"
IMPORTER="${NSYS_HOME}/host-linux-armv8/QdstrmImporter"
TARGET_DIR="${NSYS_HOME}/target-linux-sbsa-armv8"
CONTAINER_TARGET="/opt/nvidia/nsight-systems/${NSYS_VERSION}/target-linux-sbsa-armv8"

mkdir -p "${ROOT}/nsys/stats"

ensure_container_nsys() {
  sudo docker exec "${CONTAINER}" \
    mkdir -p "/opt/nvidia/nsight-systems/${NSYS_VERSION}"
  if ! sudo docker exec "${CONTAINER}" test -x "${CONTAINER_TARGET}/nsys"; then
    sudo docker cp "${TARGET_DIR}" "${CONTAINER}:${CONTAINER_TARGET}"
  fi
  sudo docker exec "${CONTAINER}" \
    ln -sf "${CONTAINER_TARGET}/nsys" /usr/local/bin/nsys
}

run_runtime() {
  local mode="$1"
  sudo docker exec "${CONTAINER}" python3 \
    "${CONTAINER_ROOT}/profile_single_request.py" \
    --mode "${mode}" \
    --execution-mode cudagraph \
    --detail runtime \
    --max-new-tokens 32 \
    --warmup 2 \
    --baseline-requests 5 \
    --profile-requests 1 \
    --output "${CONTAINER_ROOT}/${mode}.cudagraph.json"
}

run_layer_events() {
  local mode="$1"
  sudo docker exec "${CONTAINER}" python3 \
    "${CONTAINER_ROOT}/profile_single_request.py" \
    --mode "${mode}" \
    --execution-mode eager \
    --detail layer \
    --max-new-tokens 32 \
    --warmup 2 \
    --baseline-requests 3 \
    --profile-requests 1 \
    --output "${CONTAINER_ROOT}/${mode}.eager_layer.json"
}

run_graph_trace() {
  local mode="$1"
  local base="${mode}_cudagraph"
  sudo docker exec "${CONTAINER}" nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --force-overwrite=true \
    --output="${CONTAINER_ROOT}/nsys/${base}" \
    /usr/bin/python3 "${CONTAINER_ROOT}/trace_single_request.py" \
    --mode "${mode}" \
    --execution-mode cudagraph \
    --max-new-tokens 32 \
    --warmup 2 \
    --output "${CONTAINER_ROOT}/${base}.nsys_run.json"
  "${IMPORTER}" --force-overwrite \
    --input-file "${ROOT}/nsys/${base}.qdstrm" \
    --output-file "${ROOT}/nsys/${base}.nsys-rep"
  "${HOST_NSYS}" stats \
    --report cuda_gpu_kern_sum \
    --report cuda_api_sum \
    --report cuda_gpu_mem_time_sum \
    --report nvtx_gpu_proj_sum \
    --format csv \
    --output "${ROOT}/nsys/stats/${base}" \
    --force-overwrite=true \
    "${ROOT}/nsys/${base}.nsys-rep"
}

run_ptq_layer_trace() {
  local base="static_ptq_tail_eager_layer"
  sudo docker exec "${CONTAINER}" nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --force-overwrite=true \
    --output="${CONTAINER_ROOT}/nsys/${base}" \
    /usr/bin/python3 "${CONTAINER_ROOT}/trace_single_request.py" \
    --mode static_ptq_tail \
    --execution-mode eager \
    --layer-nvtx \
    --max-new-tokens 32 \
    --warmup 2 \
    --output "${CONTAINER_ROOT}/${base}.nsys_run.json"
  "${IMPORTER}" --force-overwrite \
    --input-file "${ROOT}/nsys/${base}.qdstrm" \
    --output-file "${ROOT}/nsys/${base}.nsys-rep"
  "${HOST_NSYS}" stats \
    --report nvtx_kern_sum \
    --report nvtx_gpu_proj_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --format csv \
    --output "${ROOT}/nsys/stats/${base}" \
    --force-overwrite=true \
    "${ROOT}/nsys/${base}.nsys-rep"
}

run_runtime full_fp
run_runtime static_ptq_tail
run_layer_events full_fp
run_layer_events static_ptq_tail
ensure_container_nsys
run_graph_trace full_fp
run_graph_trace static_ptq_tail
run_ptq_layer_trace
python3 "${ROOT}/analyze_profile.py" --root "${ROOT}"
sudo chown -R "$(id -u):$(id -g)" "${ROOT}"
