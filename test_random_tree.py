import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
import argparse
import torch
from torchvision import transforms
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

# Define the mean and std used in normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Reshape for broadcasting
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize(tensor):
    return tensor * std + mean

def test_tree(tree, subset):
    genome = tree_to_genome(tree)
    print(genome)
    dataset_name = "flowers102"
    num_ways = 5
    num_shots = 2
    train_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=True)
    test_path = dataset_manager.get_dataset_path(dataset_name, num_ways, num_shots, subset, train=False)

    dataset = TreeAugmentedDataset(train_path, tree, num_augmentations_per_image = 5)
    #save all the images in the dataset to a folder
    os.makedirs("augmented_images", exist_ok=True)
    os.makedirs(f"augmented_images/{str(genome)}", exist_ok=True)
    for i, (img, label) in enumerate(dataset):
        #convert the tensor to a PIL Image
        img = denormalize(img)
        img = torch.clamp(img, 0, 1)  # Ensure values are in [0,1] range
        to_pil = transforms.ToPILImage()
        img = to_pil(img)
        # img = (img * 255).byte()  # Scale to [0,255] and convert to bytes
        # img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC format
        # img = Image.fromarray(img.cpu().numpy())
        img.save(f"augmented_images/{str(genome)}/{i}.jpg")

    train_dataset = TreeAugmentedDataset(train_path, tree, num_augmentations_per_image = 5)
    test_dataset = FolderDataset(test_path)
    model = network_model.get_model_for_finetune(ModelType.RESNET50, num_ways)
    model_results: ModelResults = network_model.train_and_test(model, train_dataset, test_dataset, num_epochs = 200, device = "cuda")
    print(f"Accuracy for tree {str(genome)}, subset {subset}: {model_results.accs[-1]}")
    return model_results.accs[-1]


def main():

    random.seed(42)
    
    tree = BinaryAugmentationNode()
    tree.augmentation_type = AugmentationType.COLOR
    tree.left_probability = 0.5
    tree.left = BinaryAugmentationNode()
    tree.left.augmentation_type = AugmentationType.CLASSICAL
    tree.right = BinaryAugmentationNode()
    tree.right.augmentation_type = AugmentationType.CLASSICAL
    test_tree(tree, 47)
    test_tree(tree, 48)
    test_tree(tree, 50)

    tree = BinaryAugmentationNode()
    tree.augmentation_type = AugmentationType.COLOR
    tree.left_probability = 0.5
    tree.left = BinaryAugmentationNode()
    tree.left.augmentation_type = AugmentationType.COLOR
    tree.left.left_probability = 0.5
    tree.left.left = BinaryAugmentationNode()
    tree.left.left.augmentation_type = AugmentationType.CLASSICAL
    tree.left.right = BinaryAugmentationNode()
    tree.left.right.augmentation_type = AugmentationType.CLASSICAL
    tree.right = BinaryAugmentationNode()
    tree.right.augmentation_type = AugmentationType.COLOR
    tree.right.left_probability = 0.5
    tree.right.left = BinaryAugmentationNode()
    tree.right.left.augmentation_type = AugmentationType.CLASSICAL
    tree.right.right = BinaryAugmentationNode()
    tree.right.right.augmentation_type = AugmentationType.CLASSICAL
    test_tree(tree, 47)
    test_tree(tree, 48)
    test_tree(tree, 50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()