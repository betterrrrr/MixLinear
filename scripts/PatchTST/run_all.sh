#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=0
REQUIRED_MEM_MIB=12000

scripts=(
  "electricity.sh"
  "traffic.sh"
  "ettm1.sh"
  "ettm2.sh"
  "etth1.sh"
  "etth2.sh"
)

echo "Running PatchTST scripts from $SCRIPT_DIR"

echo "GPU target: $GPU, required free memory: ${REQUIRED_MEM_MIB} MiB"

query_gpu_free_mem() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '
}

wait_for_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found; skipping GPU free memory check."
    return
  fi

  while true; do
    free_mem=$(query_gpu_free_mem)
    if [ -z "$free_mem" ]; then
      echo "ERROR: unable to query GPU free memory."
      exit 1
    fi
    if [ "$free_mem" -ge "$REQUIRED_MEM_MIB" ]; then
      echo "GPU${GPU} free memory ${free_mem} MiB >= ${REQUIRED_MEM_MIB} MiB. Starting next script."
      break
    fi
    echo "GPU${GPU} free memory ${free_mem} MiB < required ${REQUIRED_MEM_MIB} MiB. Waiting 30s..."
    sleep 30
  done
}

for script in "${scripts[@]}"; do
  script_path="$SCRIPT_DIR/$script"
  if [ ! -f "$script_path" ]; then
    echo "Missing script: $script_path"
    exit 1
  fi
  wait_for_gpu
  echo "---- Running $script ----"
  bash "$script_path"
  echo "---- Finished $script ----"
  echo
done
