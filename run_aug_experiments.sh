#!/bin/bash

AUGMENTATIONS=("--use_canny" "--use_depth" "--use_seg" "--use_color" "--use_nerf")
EPOCHS=(128)
BATCH_SIZES=(16)
LEARNING_RATE=(0.0003 0.0001)

run_experiment() {
    aug=$1
    epochs=$2
    batch_size=$3
    learning_rate=$4

    echo "Running experiment: $aug, epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate (Run $run)"
    python train.py $aug \
        --epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --architecture resnet50
}

for aug in "${AUGMENTATIONS[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for learning_rate in "${LEARNING_RATE[@]}"; do
                for run in {1..3}; do
                    run_experiment "$aug" "$epochs" "$batch_size" "$learning_rate" "$run"
                    sleep 5
                done
            done
        done
    done
done