#!/bin/bash
# Create datasets for Caltech256

datasets=("stanford_cars" "flowers102" "stanford_dogs" "caltech256" "food101")

# python dataset_manager.py --dataset caltech256 --subset 43 --num_ways 5 --num_shots 1
# python dataset_manager.py --dataset caltech256 --subset 42 --num_ways 5 --num_shots 2
# python dataset_manager.py --dataset caltech256 --subset 42 --num_ways 5 --num_shots 5
# python dataset_manager.py --dataset caltech256 --subset 42 --num_ways 10 --num_shots 2
# python dataset_manager.py --dataset caltech256 --subset 42 --num_ways 10 --num_shots 2

python dataset_manager.py --dataset flowers102 --subset 50 --num_ways 5 --num_shots 1
python dataset_manager.py --dataset flowers102 --subset 50 --num_ways 5 --num_shots 2
python dataset_manager.py --dataset flowers102 --subset 50 --num_ways 5 --num_shots 5
python dataset_manager.py --dataset flowers102 --subset 48 --num_ways 10 --num_shots 2
python dataset_manager.py --dataset flowers102 --subset 48 --num_ways 10 --num_shots 5

python dataset_manager.py --dataset stanford_dogs --subset 46 --num_ways 5 --num_shots 1
python dataset_manager.py --dataset stanford_dogs --subset 47 --num_ways 5 --num_shots 2
python dataset_manager.py --dataset stanford_dogs --subset 47 --num_ways 5 --num_shots 5
python dataset_manager.py --dataset stanford_dogs --subset 45 --num_ways 10 --num_shots 2
python dataset_manager.py --dataset stanford_dogs --subset 45 --num_ways 10 --num_shots 5

python dataset_manager.py --dataset stanford_cars --subset 44 --num_ways 5 --num_shots 1
python dataset_manager.py --dataset stanford_cars --subset 44 --num_ways 5 --num_shots 2
python dataset_manager.py --dataset stanford_cars --subset 44 --num_ways 5 --num_shots 5
python dataset_manager.py --dataset stanford_cars --subset 48 --num_ways 10 --num_shots 2
python dataset_manager.py --dataset stanford_cars --subset 48 --num_ways 10 --num_shots 5


python dataset_manager.py --dataset oxford-iiit-pet --subset 44 --num_ways 5 --num_shots 1
python dataset_manager.py --dataset oxford-iiit-pet --subset 44 --num_ways 5 --num_shots 2
python dataset_manager.py --dataset oxford-iiit-pet --subset 41 --num_ways 5 --num_shots 5
python dataset_manager.py --dataset oxford-iiit-pet --subset 45 --num_ways 10 --num_shots 2
python dataset_manager.py --dataset oxford-iiit-pet --subset 45 --num_ways 10 --num_shots 5


# python dataset_manager.py --dataset food101 --subset 43 --num_ways 5 --num_shots 1
# python dataset_manager.py --dataset food101 --subset 43 --num_ways 5 --num_shots 2
# python dataset_manager.py --dataset food101 --subset 43 --num_ways 5 --num_shots 5
# python dataset_manager.py --dataset food101 --subset 42 --num_ways 10 --num_shots 2
# python dataset_manager.py --dataset food101 --subset 43 --num_ways 10 --num_shots 5



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
