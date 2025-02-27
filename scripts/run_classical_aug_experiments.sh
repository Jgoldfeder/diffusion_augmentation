#!/bin/bash

# Array of seeds to use
subsets=(48 49 50)
datasets=("caltech256" "stanford_cars")
shots=(2)
seeds=(41 43)
# For each seed, run the experiment once
for subset in "${subsets[@]}"; do
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running experiment with subset $subset" "ways 5" "shots $shot" "dataset $dataset" "seed $seed"
                python test_classical_aug_baseline.py --subset $subset --shots $shot --dataset $dataset --ways 5 --seed $seed
            done
        done
    done
    echo "Completed experiment with subset $subset"
    echo "-----------------------------------"
done

echo "All experiments completed!"
