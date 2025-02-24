import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import wandb
import pygad
import logging
import argparse
import torch
import numpy as np
from PIL import Image
import network_model
import dataset_manager
from network_model import ModelResults, ModelType
from dataset_manager import FolderDataset
from augmentation_tree import BinaryAugmentationNode, AugmentationType, TreeAugmentedDataset, ProbabilityLimits
import time

def tree_to_genome(tree):
    genome = []
    q = []
    q.append(tree)
    while len(q) > 0:
        node = q.pop(0)
        genome.append(node.augmentation_type.value)
        genome.append(round(node.left_probability, 2))
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return genome
def main():

    random.seed(time.time())
    
    tree = BinaryAugmentationNode()
    tree.make_random_tree(3)
    genome = tree_to_genome(tree)
    print(genome)

    dataset_name = "flowers102"
    num_ways = 5
    num_shots = 2
    subset = 50
    train_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=True)
    test_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=False)

    dataset = TreeAugmentedDataset(train_path, tree, num_augmentations_per_image = 5)
    #save all the images in the dataset to a folder
    os.makedirs("augmented_images", exist_ok=True)
    os.makedirs(f"augmented_images/{str(genome)}", exist_ok=True)
    for i, (img, label) in enumerate(dataset):
        #convert the tensor to a PIL Image
        img = torch.clamp(img, 0, 1)  # Ensure values are in [0,1] range
        img = (img * 255).byte()  # Scale to [0,255] and convert to bytes
        img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC format
        img = Image.fromarray(img.cpu().numpy())
        img.save(f"augmented_images/{str(genome)}/{i}.jpg")
    train_dataset, val_dataset = dataset_manager.split_train_val(dataset)

    model = network_model.get_model_for_finetune(ModelType.RESNET50, num_ways)
    model_results: ModelResults = network_model.train_and_val(model, train_dataset, val_dataset, num_iterations_for_val = 10, device = "cuda")


if __name__ == "__main__":
    main()