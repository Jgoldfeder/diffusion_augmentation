#!/bin/bash

DATASETS=("caltech256")
AUGMENTATIONS=(
    "--use_canny --use_depth --use_seg --use_color --use_nerf"
)

#  "--use_depth --use_seg --use_color --use_nerf"
#     "--use_nerf"
#     "--use_canny"
#     "--use_depth"
#     "--use_seg"
#     "--use_color"
#     "--use_canny --use_depth --use_seg"
#     "--use_canny --use_depth --use_color"
#     "--use_canny --use_seg --use_color"
#     "--use_depth --use_seg --use_color"

EPOCHS=(400)
BATCH_SIZES=(32 128)
LEARNING_RATE=(0.0003)

#FOR LATER:
#NUM_CLASSES = (5, 10)
#IMAGES_PER_CLASS = (1, 2)

run_experiment() {
    dataset=$1
    aug=$2
    epochs=$3
    batch_size=$4
    learning_rate=$5

    echo "Running experiment: dataset=$dataset, $aug, epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate (Seed $seed)"
    python train.py $aug \
        --dataset $dataset \
        --epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --architecture resnet50 \
        --seed $seed
}

for dataset in "${DATASETS[@]}"; do
    for aug in "${AUGMENTATIONS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for learning_rate in "${LEARNING_RATE[@]}"; do
                    for seed in {41..43}; do
                        run_experiment "$dataset" "$aug" "$epochs" "$batch_size" "$learning_rate" "$seed"
                        sleep 5
                    done
                done
            done
        done
    done
done
