#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

scripts=(
  "scripts/MixLinear/electricity.sh"
  "scripts/MixLinear/etth1.sh"
  "scripts/MixLinear/etth2.sh"
  "scripts/MixLinear/ettm1.sh"
  "scripts/MixLinear/ettm2.sh"
  "scripts/MixLinear/traffic.sh"
  "scripts/MixLinear/weather.sh"
)

for script in "${scripts[@]}"; do
  echo "Running $script"
  bash "$script"
done

echo "All done."
