#!/bin/bash

# Function to run experiments for a specific model
run_experiments() {
    local model_type=$1
    
    # Flowers102 Experiments
    python test_random_tree.py --dataset flowers102 --num_ways 5 --num_shots 1 --model_type $model_type --subset 50
    python test_random_tree.py --dataset flowers102 --num_ways 5 --num_shots 2 --model_type $model_type --subset 50
    python test_random_tree.py --dataset flowers102 --num_ways 5 --num_shots 5 --model_type $model_type --subset 50
    python test_random_tree.py --dataset flowers102 --num_ways 10 --num_shots 2 --model_type $model_type --subset 48
    python test_random_tree.py --dataset flowers102 --num_ways 10 --num_shots 5 --model_type $model_type --subset 48

    # Stanford Dogs Experiments
    python test_random_tree.py --dataset stanford_dogs --num_ways 5 --num_shots 1 --model_type $model_type --subset 46
    python test_random_tree.py --dataset stanford_dogs --num_ways 5 --num_shots 2 --model_type $model_type --subset 47
    python test_random_tree.py --dataset stanford_dogs --num_ways 5 --num_shots 5 --model_type $model_type --subset 47
    python test_random_tree.py --dataset stanford_dogs --num_ways 10 --num_shots 2 --model_type $model_type --subset 45
    python test_random_tree.py --dataset stanford_dogs --num_ways 10 --num_shots 5 --model_type $model_type --subset 45

    # Stanford Cars Experiments
    python test_random_tree.py --dataset stanford_cars --num_ways 5 --num_shots 1 --model_type $model_type --subset 44
    python test_random_tree.py --dataset stanford_cars --num_ways 5 --num_shots 2 --model_type $model_type --subset 44
    python test_random_tree.py --dataset stanford_cars --num_ways 5 --num_shots 5 --model_type $model_type --subset 44
    python test_random_tree.py --dataset stanford_cars --num_ways 10 --num_shots 2 --model_type $model_type --subset 48
    python test_random_tree.py --dataset stanford_cars --num_ways 10 --num_shots 5 --model_type $model_type --subset 48

    # Oxford-IIIT-Pet Experiments
    python test_random_tree.py --dataset oxford-iiit-pet --num_ways 5 --num_shots 1 --model_type $model_type --subset 44
    python test_random_tree.py --dataset oxford-iiit-pet --num_ways 5 --num_shots 2 --model_type $model_type --subset 44
    python test_random_tree.py --dataset oxford-iiit-pet --num_ways 5 --num_shots 5 --model_type $model_type --subset 41
    python test_random_tree.py --dataset oxford-iiit-pet --num_ways 10 --num_shots 2 --model_type $model_type --subset 45
    python test_random_tree.py --dataset oxford-iiit-pet --num_ways 10 --num_shots 5 --model_type $model_type --subset 45
}

# Run experiments for all models
echo "Running experiments for ResNet50..."
run_experiments "resnet50"

echo "Running experiments for ViT-Small..."
run_experiments "vits"

echo "Running experiments for MobileNetV2..."
run_experiments "mobilenetv2" 