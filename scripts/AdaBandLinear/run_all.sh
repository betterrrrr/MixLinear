#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

scripts=(
    "$SCRIPT_DIR/etth1.sh"
    "$SCRIPT_DIR/etth2.sh"
    "$SCRIPT_DIR/ettm1.sh"
    "$SCRIPT_DIR/ettm2.sh"
    "$SCRIPT_DIR/electricity.sh"
    "$SCRIPT_DIR/weather.sh"
    "$SCRIPT_DIR/traffic.sh"
)

for script in "${scripts[@]}"; do
    echo ">>> Running $script"
    bash "$script"
done
