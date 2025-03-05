#!/bin/bash

# Array of seeds to use
#subsets=(41 42 43 44 45 46 47 48 49 50)
datasets=("stanford_cars" "flowers102" "caltech256" "stanford_dogs" "food101")
shots=(1)
caltech_subsets=(42 43 48)
flowers_subsets=(47 48 50)
stanford_dogs_subsets=(41 46 47)
stanford_cars_subsets=(44 46 47)
food101_subsets=(41 42 43)

seeds=(41 42 43)

for subset in "${caltech_subsets[@]}"; do
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running experiment for $dataset with $shot shots and seed $seed"
                python run_classical_aug_experiments.py --dataset $dataset --shots $shot --seed $seed
            done
        done
    done
done

for subset in "${flowers_subsets[@]}"; do
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running experiment for $dataset with $shot shots and seed $seed"
                python run_classical_aug_experiments.py --dataset $dataset --shots $shot --seed $seed
            done
        done
    done
done

for subset in "${stanford_dogs_subsets[@]}"; do
    for dataset in "${datasets[@]}"; do 
        for shot in "${shots[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running experiment for $dataset with $shot shots and seed $seed"
                python run_classical_aug_experiments.py --dataset $dataset --shots $shot --seed $seed
            done
        done
    done
done

for subset in "${stanford_cars_subsets[@]}"; do
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Running experiment for $dataset with $shot shots and seed $seed"
                python run_classical_aug_experiments.py --dataset $dataset --shots $shot --seed $seed
            done
        done
    done
done



# # For each seed, run the experiment once
# for subset in "${subsets[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         for shot in "${shots[@]}"; do
#             for seed in "${seeds[@]}"; do
#                 echo "Running experiment with subset $subset" "ways 5" "shots $shot" "dataset $dataset" "seed $seed"
#                 python test_classical_aug_baseline.py --subset $subset --shots $shot --dataset $dataset --ways 5 --seed $seed
#             done
#         done
#     done
#     echo "Completed experiment with subset $subset"
#     echo "-----------------------------------"
# done

echo "All experiments completed!"
