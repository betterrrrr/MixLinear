wait_for_gpu_memory() {
  local gpu_id="${ITRANSFORMER_GPU_ID:-${CUDA_VISIBLE_DEVICES:-3}}"
  local min_free_mb="${MIN_FREE_MEM_MB:-4000}"
  local sleep_seconds="${GPU_WAIT_SECONDS:-60}"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" | head -n1 | tr -d '[:space:]')"
    if [ -n "$free_mb" ] && [ "$free_mb" -ge "$min_free_mb" ] 2>/dev/null; then
      return 0
    fi
    echo "GPU $gpu_id free ${free_mb:-unknown} MiB, waiting ${sleep_seconds}s..."
    sleep "$sleep_seconds"
  done
}

run_with_gpu_retry() {
  local log_file="$1"
  shift

  while true; do
    wait_for_gpu_memory
    if "$@" >"$log_file" 2>&1; then
      return 0
    fi

    if grep -Eq 'OutOfMemoryError|CUDA out of memory|CUDNN_STATUS_INTERNAL_ERROR' "$log_file"; then
      echo "GPU memory pressure detected for $log_file, retrying after ${GPU_WAIT_SECONDS:-60}s..."
      sleep "${GPU_WAIT_SECONDS:-60}"
      continue
    fi

    return 1
  done
}

setup_iTransformer_env() {
  if [ ! -d "./logs" ]; then
    mkdir ./logs
  fi

  if [ -x "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi

  export PYTHONNOUSERSITE=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
  export ITRANSFORMER_GPU_ID="${ITRANSFORMER_GPU_ID:-3}"
}