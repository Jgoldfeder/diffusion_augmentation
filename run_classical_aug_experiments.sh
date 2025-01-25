#!/bin/bash

# Array of seeds to use
seeds=(41 42 43 44 45 46 47 48 49 50)
datasets=("flowers102" "caltech256")
shots=(5 10)

# For each seed, run the experiment once
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            echo "Running experiment with seed $seed" "ways 5" "shots $shot" "dataset $dataset" 
            python test_dataset.py --seed $seed --shots $shot --dataset $dataset --ways 5
        done
    done
    echo "Completed experiment with seed $seed"
    echo "-----------------------------------"
done

echo "All experiments completed!"
