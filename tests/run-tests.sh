#!/bin/bash
set -e

cd "$(dirname "$0")"

# Split dataset1
siin-trainer split-dataset --dataset ./dataset1 --val 0.25 --test 0.25 --seed 1

# Merge dataset1 & dataset2
siin-trainer merge-datasets --datasets ./dataset1  --datasets  ./dataset2 --output ./merged_dataset