#!/bin/bash

# Array of seeds to use
seeds=(44 45 46 47 48 49 50)

# For each seed, run the experiment once
for seed in "${seeds[@]}"; do
    echo "Running experiment with seed $seed"
    python test_dataset.py --seed $seed --shots 2 --dataset caltech256 --ways 5
    echo "Completed experiment with seed $seed"
    echo "-----------------------------------"
done

echo "All experiments completed!"
