import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import wandb
import pygad
import logging
import argparse
import torch
import numpy as np

import network_model
import dataset_manager
from network_model import ModelResults, ModelType
from dataset_manager import FolderDataset
from augmentation_tree import BinaryAugmentationNode, AugmentationType, TreeAugmentedDataset, ProbabilityLimits
import time

def tree_to_string(node, level=0, direction='root'):
    tree_str = ''
    if node:
        if not node.left and not node.right:
            # Leaf nodes: only show augmentation type
            edge_info = f"(Augmentation: {node.augmentation_type})"
        else:
            # Non-leaf nodes: show augmentation type and probabilities
            edge_info = f"(Augmentation: {node.augmentation_type}, L_prob: {node.left_probability:.2f}, R_prob: {1-node.left_probability:.2f})"
        tree_str += '  ' * level + f"{direction}: {edge_info}" + '\n'
        if node.left:
            tree_str += tree_to_string(node.left, level + 1, 'L')
        if node.right:
            tree_str += tree_to_string(node.right, level + 1, 'R')
    return tree_str

def main():

    random.seed(time.time())
    
    tree = BinaryAugmentationNode()
    tree.make_random_tree(3)
    print(tree_to_string(tree))

    dataset_name = "flowers102"
    num_ways = 5
    num_shots = 2
    subset = 50
    train_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=True)
    test_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=False)

    dataset = TreeAugmentedDataset(train_path, tree, num_augmentations_per_image = 5)
    #save all the images in the dataset to a folder
    os.makedirs("augmented_images", exist_ok=True)
    os.makedirs(f"augmented_images/{tree_to_string(tree)}", exist_ok=True)
    for i, (img, label) in enumerate(dataset):
        img.save(f"augmented_images/{tree_to_string(tree)}/{i}.jpg")
    train_dataset, val_dataset = dataset_manager.split_train_val(dataset)

    model = network_model.get_model_for_finetune(ModelType.RESNET18, num_ways)
    model_results: ModelResults = network_model.train_and_val(model, train_dataset, val_dataset, num_iterations_for_val = 10, device = "cuda")


if __name__ == "__main__":
    main()