#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

scripts=(
    "$SCRIPT_DIR/etth1.sh"
)

for script in "${scripts[@]}"; do
    bash "$script"
done
