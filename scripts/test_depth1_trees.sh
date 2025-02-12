#!/bin/bash
for seed in {42..50}; do
    python test_depth1_trees.py --dataset caltech256 --seed $seed --num_shots 2 --num_ways 5
done