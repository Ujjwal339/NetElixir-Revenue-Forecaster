#!/bin/bash

set -e

DATA_DIR=$1
MODEL_PATH=$2
OUTPUT_PATH=$3

echo "========================================="
echo "NetElixir Revenue Forecaster"
echo "========================================="

echo
echo "Generating features..."

python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --output features.parquet

echo
echo "Running prediction..."

python src/predict.py \
    --features features.parquet \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo
echo "Done!"
echo "Predictions saved to:"
echo "$OUTPUT_PATH"
