#!/bin/bash

for alpha in $(seq 0 0.1 1); do
    python genetic_algorithm.py \
        --num_generations 10 \
        --sol_per_pop 14 \
        --dataset stanford_cars \
        --num_ways 5 \
        --num_shots 1 \
        --subset 44 \
        --model_type resnet50 \
        --tree_depth 2 \
        --num_augmentations_per_image 2 \
        --num_iterations_for_test 400 \
        --seed 42 \
        --one_shot_clustering True \
        --clustering_alpha $alpha
done