import wandb
import argparse
from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode, AugmentationType, ProbabilityLimits
from network_model import ModelType
import os
import logging
import dataset_manager
import network_model
from genetic_algorithm import genome_to_tree
import torch
import random
import time
from torchvision import transforms

class TreeAugmentedDatasetWithClassical(TreeAugmentedDataset):
    def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
        super().__init__(dataset_path, augmentation_tree, num_augmentations_per_image)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return dataset_manager.get_base_transform()(dataset_manager.get_classical_transform()(img)), label

def generate_random_genome():
    genome = []
    for i in range(3):  # We need 3 pairs of numbers
        # First number (0-7)
        genome.append(random.randint(0, 7))
        # Second number (0.3-0.7)
        genome.append(round(random.uniform(0.3, 0.7), 2))
    return genome

def parse_args():
    parser = argparse.ArgumentParser(description='Test random tree accuracy with different models and datasets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., oxford-iiit-pet, caltech256)')
    parser.add_argument('--num_ways', type=int, required=True, help='Number of ways (classes)')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots (examples per class)')
    parser.add_argument('--model_type', type=str, required=True, 
                      choices=['resnet50', 'vit224', 'mobilenetv2', 'vits'], 
                      help='Model type to use (resnet50: standard CNN, vit224: Vision Transformer, mobilenetv2: lightweight CNN, vits: small Vision Transformer)')
    parser.add_argument('--subset', type=int, default=44, help='Subset of classes to use')
    parser.add_argument('--num_augmentations', type=int, default=2, help='Number of augmentations per image')
    parser.add_argument('--num_runs', type=int, default=6, help='Number of runs to perform')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--seed_start', type=int, default=41, help='Starting seed for random number generation')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Generate random genome
    tree_genome = generate_random_genome()
    print(f"Generated random genome: {tree_genome}")

    node = genome_to_tree(tree_genome)
    print(f"Using augmentation tree: {str(node)}")

    train_path = dataset_manager.get_dataset_path(args.dataset, args.num_ways, args.num_shots, args.subset, train=True)
    test_path = dataset_manager.get_dataset_path(args.dataset, args.num_ways, args.num_shots, args.subset, train=False)

    for i in range(args.num_runs):
        seed = args.seed_start + i
        wandb.init(
            project="random-tree-tests",
            config={
                "subset": args.subset,
                "num_shots": args.num_shots,
                "dataset": args.dataset,
                "num_ways": args.num_ways,
                "model_type": args.model_type,
                "seed": seed,
                "without_classical": i % 2,
                "genome": tree_genome
            }
        )
        random.seed(seed)

        if i % 2:
            train_dataset = TreeAugmentedDataset(train_path, node, args.num_augmentations)
        else:
            train_dataset = TreeAugmentedDatasetWithClassical(train_path, node, args.num_augmentations)
        test_dataset = FolderDataset(test_path)

        model = network_model.get_model_for_finetune(ModelType(args.model_type), args.num_ways)
        model_results = network_model.train_and_test(model, train_dataset, test_dataset, args.num_epochs, 'cuda')
        print(f"Run {i+1}/{args.num_runs} Results:")
        print(model_results)

        with open('temp.txt', 'a') as f:
            f.write(f"Run {i+1} - {'without' if i % 2 else 'with'} classical augmentation\n")
            f.write(f"Model: {args.model_type}, Dataset: {args.dataset}, Ways: {args.num_ways}, Shots: {args.num_shots}\n")
            f.write(f"Genome: {tree_genome}\n")
            f.write(str(model_results.accs[-6:]) + '\n')

        results = model_results
        for (train_loss, train_acc, test_loss, test_acc) in zip(results.train_losses, results.train_accs, results.losses, results.accs):
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc
            })
        wandb.log({
            'tree': str(node),
            'model_type': args.model_type,
            'genome': tree_genome,
            'best_test_accuracy': max(results.accs)
        })
        wandb.finish()