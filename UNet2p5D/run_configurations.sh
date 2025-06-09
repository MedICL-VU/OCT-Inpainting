#!/bin/bash

BASE_CMD="python3 run_inpainting_pipeline.py"

EXPERIMENTS=(
  "--stack_size 5 --file_name batchsize4"
  "--stack_size 7 --file_name stacksize7"
  "--stack_size 11 --file_name stacksize11"
  "--stack_size 13 --file_name batchsize4"
  "--lr 1e-5 --file_name lr1e-5"
  "--lr 1e-4 --file_name lr1e-4"
  "--augment --file_name augment"
)

for i in "${!EXPERIMENTS[@]}"; do
  echo "=== Running experiment $((i+1))/9: ${EXPERIMENTS[$i]} ==="
  $BASE_CMD ${EXPERIMENTS[$i]}
done