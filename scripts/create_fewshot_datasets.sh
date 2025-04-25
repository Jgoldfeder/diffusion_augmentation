#!/bin/bash
# Create datasets for Caltech256

datasets=("stanford_cars" "flowers102" "stanford_dogs" "caltech256" "food101")


python dataset_manager.py --dataset food101 --subset 42 --num_ways 10 --num_shots 2

for dataset in "${datasets[@]}"; do
    for subset in {41..50}; do
        echo "Creating dataset for $dataset with subset $subset num_ways 10 num_shots 2"
        python dataset_manager.py --dataset $dataset --subset $subset --num_ways 10 --num_shots 2
    done
done

# for subset in {41..50}; do
#     python dataset_manager.py --dataset flowers102 --subset $subset --num_ways 5 --num_shots 1
# done

# for subset in {41..50}; do
#     python dataset_manager.py --dataset caltech256 --subset $subset --num_ways 5 --num_shots 1
# done

# for subset in {41..50}; do
#     python dataset_manager.py --dataset stanford_dogs --subset $subset --num_ways 5 --num_shots 1
# done

# for subset in {41..50}; do
#     python dataset_manager.py --dataset food101 --subset $subset --num_ways 5 --num_shots 1
# done

# for subset in {41..50}; do
#     for shot in 5 10; do
#         echo "Creating dataset for Caltech256 with subset $subset and shot $shot"
#         python dataset_manager.py --dataset caltech256 --subset $subset --num_ways 5 --num_shots $shot
#     done
# done

# # Create datasets for Flowers102
# for subset in {41..50}; do
#     for shot in 5 10; do
#         echo "Creating dataset for Flowers102 with subset $subset and shot $shot"
#         python dataset_manager.py --dataset flowers102 --subset $subset --num_ways 5 --num_shots $shot
#     done
# done
