#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for script in electricity.sh etth1.sh etth2.sh ettm1.sh ettm2.sh traffic.sh weather.sh; do
  echo "========== Running $script =========="
  bash "$SCRIPT_DIR/$script"
  echo "========== Finished $script =========="
  echo
 done
