#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Generating features..."

python src/generate_features.py \
    --mode inference \
    --data-dir "$DATA_DIR" \
    --output features.parquet

echo "Running prediction..."

python src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo
echo "Done."
echo
echo "Predictions written to:"
echo "$OUTPUT_PATH"
