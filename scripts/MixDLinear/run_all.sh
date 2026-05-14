#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Starting MixDLinear experiments for all 7 datasets..."

echo "==================== Running ETTh1 ===================="
bash ./etth1.sh

echo "==================== Running ETTh2 ===================="
bash ./etth2.sh

echo "==================== Running ETTm1 ===================="
bash ./ettm1.sh

echo "==================== Running ETTm2 ===================="
bash ./ettm2.sh

echo "==================== Running Electricity ===================="
bash ./electricity.sh

echo "==================== Running Traffic ===================="
bash ./traffic.sh

echo "==================== Running Weather ===================="
bash ./weather.sh

echo "All 7 datasets have been completed!"
