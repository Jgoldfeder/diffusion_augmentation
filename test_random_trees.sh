#!/bin/bash

# Datasets to test

# Number of shots to test
shots=(5 10)

flowers102_seeds=(47 48 50)
caltech_seeds=(42 43 48)

for shot in "${shots[@]}"; do
    for seed in "${caltech_seeds[@]}"; do
        echo "Running test for caltech256, ${shot}-shot, seed $seed"
        python test_random_tree.py --dataset caltech256 --seed $seed --num_ways 5 --num_shots $shot
    done
done

for shot in "${shots[@]}"; do
    for seed in "${flowers102_seeds[@]}"; do
        echo "Running test for flowers102, ${shot}-shot, seed $seed"
        python test_random_tree.py --dataset flowers102 --seed $seed --num_ways 5 --num_shots $shot
    done
done

echo "All tests completed!"
