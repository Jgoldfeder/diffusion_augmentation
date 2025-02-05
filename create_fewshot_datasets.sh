#!/bin/bash
# Create datasets for Caltech256

for subset in {41..50}; do
    python create_dataset.py --dataset flowers102 --subset $subset --num_ways 5 --num_shots 2
done

for subset in {41..50}; do
    python create_dataset.py --dataset flowers102 --subset $subset --num_ways 5 --num_shots 2
done

# for seed in {41..50}; do
#     for shot in 5 10; do
#         echo "Creating dataset for Caltech256 with seed $seed and shot $shot"
#         python create_dataset.py --dataset caltech256 --seed $seed --num_ways 5 --num_shots $shot
#     done
# done

# # Create datasets for Flowers102
# for seed in {41..50}; do
#     for shot in 5 10; do
#         echo "Creating dataset for Flowers102 with seed $seed and shot $shot"
#         python create_dataset.py --dataset flowers102 --seed $seed --num_ways 5 --num_shots $shot
#     done
# done