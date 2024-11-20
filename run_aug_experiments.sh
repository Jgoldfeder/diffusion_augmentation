#!/bin/bash

DATASETS=("flowers102" "caltech256")
AUGMENTATIONS=("--use_canny" "--use_depth" "--use_seg" "--use_color" "--use_nerf")
EPOCHS=(128)
BATCH_SIZES=(16)
LEARNING_RATE=(0.0003 0.0001)

run_experiment() {
    dataset=$1
    aug=$2
    epochs=$3
    batch_size=$4
    learning_rate=$5

    echo "Running experiment: dataset=$dataset, $aug, epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate (Run $run)"
    python train.py $aug \
        --dataset $dataset \
        --epochs $epochs \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --architecture resnet50
}

for dataset in "${DATASETS[@]}"; do
    for aug in "${AUGMENTATIONS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for learning_rate in "${LEARNING_RATE[@]}"; do
                    for run in {1..3}; do
                        run_experiment "$dataset" "$aug" "$epochs" "$batch_size" "$learning_rate" "$run"
                        sleep 5
                    done
                done
            done
        done
    done
done