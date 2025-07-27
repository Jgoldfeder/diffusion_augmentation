import torchvision.transforms as transforms
import argparse
from dataset_manager import FolderDataset, get_dataset_path
from network_model import train_and_test
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch import nn
import torch
import wandb
import os
import random
import time
import logging
from network_model import get_model_for_finetune, ModelType
from torchvision.transforms import RandAugment

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

class RandAugmentDataset(FolderDataset):
    def __init__(self, dataset_path, N=2, M=10, duplicate_factor=6):
        super().__init__(dataset_path)
        self.duplicate_factor = duplicate_factor
        self.N = N
        self.M = M
        
        # Create RandAugment transform with specified parameters
        self.randaugment_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop(size=(224, 224)),
            RandAugment(num_ops=N, magnitude=M),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create base transform for original images
        self.base_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images) * self.duplicate_factor

    def __getitem__(self, index):
        true_index = index % len(self.images)
        img = self.images[true_index]
        label = self.labels[true_index]
        
        # Apply RandAugment transform
        transformed_img = self.randaugment_transform(img)
        return transformed_img, label

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Add argument parsing
    parser = argparse.ArgumentParser(description='RandAugment few-shot learning training script')
    parser.add_argument('--subset', type=int, default=41, help='Random subset')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset name')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'mobilenet', 'vit'], 
                       help='Model architecture to use')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Map model argument to ModelType
    model_mapping = {
        'resnet': ModelType.RESNET50,
        'mobilenet': ModelType.MOBILENETV2,
        'vit': ModelType.VIT224
    }
    model_type = model_mapping[args.model]

    # Grid search parameters
    N_values = [1, 2, 3]
    M_values = [5, 10, 15]
    
    # Run experiments for each configuration
    for N in N_values:
        for M in M_values:
            print(f"\n=== Running experiment with N={N}, M={M} ===")
            
            # Initialize wandb for this configuration
            wandb.init(
                project="randaugment-experiments",
                config={
                    "subset": args.subset,
                    "shots": args.shots,
                    "dataset": args.dataset,
                    "ways": args.ways,
                    "learning_rate": 0.001, 
                    "batch_size": 32,
                    "model": args.model,
                    "seed": args.seed,
                    "N": N,
                    "M": M,
                    "augmentation": "randaugment"
                },
                name=f"randaugment_{args.model}_N{N}_M{M}_subset{args.subset}"
            )
            
            random.seed(args.seed)

            train_path = get_dataset_path(args.dataset, args.ways, args.shots, args.subset, train=True)
            test_path = get_dataset_path(args.dataset, args.ways, args.shots, args.subset, train=False)
            
            # Create datasets with current N, M configuration
            train_dataset = RandAugmentDataset(train_path, N=N, M=M, duplicate_factor=6)
            test_dataset = FolderDataset(test_path)

            model = get_model_for_finetune(model_type, args.ways)
            results = train_and_test(model, train_dataset, test_dataset, num_epochs=200, device=device)

            for (train_loss, train_acc, test_loss, test_acc) in zip(results.train_losses, results.train_accs, results.losses, results.accs):
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                })

            wandb.finish()
            print(f"Completed experiment with N={N}, M={M}")
