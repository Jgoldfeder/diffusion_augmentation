#!/bin/bash

# Datasets to test

# Number of shots to test
shots=(2)

flowers102_subsets=(47 48 50)
caltech_subsets=(42 43 48)

for shot in "${shots[@]}"; do
    for subset in "${caltech_subsets[@]}"; do
        for run in {1..3}; do
            echo "Running test for caltech256, ${shot}-shot, subset $subset, run $run"
            python test_random_tree.py --dataset caltech256 --subset $subset --num_ways 5 --num_shots $shot
        done
    done
done

# for shot in "${shots[@]}"; do
#     for subset in "${flowers102_subsets[@]}"; do
#         for run in {1..3}; do
#             echo "Running test for flowers102, ${shot}-shot, subset $subset, run $run"
#             python test_random_tree.py --dataset flowers102 --subset $subset --num_ways 5 --num_shots $shot
#         done
#     done
# done

echo "All tests completed!"
