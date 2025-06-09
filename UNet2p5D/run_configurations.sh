#!/bin/bash

BASE_CMD="python3 run_inpainting_pipeline.py --num_runs 3"

EXPERIMENTS=(
  "--stack_size 7 --file_name stacksize7"
  "--stack_size 11 --file_name stacksize11"
  "--stack_size 13 --file_name stacksize13"
  "--lr 1e-5 --file_name lr1e-5"
  "--lr 1e-4 --file_name lr1e-4"
  "--augment --file_name augment"
  "--static_corruptions --file_name static_corruptions"
)

for i in "${!EXPERIMENTS[@]}"; do
  echo "=== Running experiment $((i+1))/7: ${EXPERIMENTS[$i]} ==="
  $BASE_CMD ${EXPERIMENTS[$i]}
done