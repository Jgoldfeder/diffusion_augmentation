import torchvision.transforms as transforms
import argparse
from dataset_manager import ClassicalDataset, FolderDataset, get_dataset_path
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Few-shot learning training script')
    parser.add_argument('--subset', type=int, default=41, help='Random subset')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset name')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    wandb.init(
        project="classical-augmentation-experiments",
        config={
            "subset": args.subset,
            "shots": args.shots,
            "dataset": args.dataset,
            "ways": args.ways,
            "learning_rate": 0.001, 
            "batch_size": 32,
            "model": "resnet50",
            "seed": args.seed
        }
    )
    
    random.seed(args.seed)

    train_path = get_dataset_path(args.dataset, args.ways, args.shots, args.subset, train=True)
    test_path = get_dataset_path(args.dataset, args.ways, args.shots, args.subset, train=False)
    train_dataset = FolderDataset(train_path)
    test_dataset = FolderDataset(test_path)

    classical_dataset = ClassicalDataset(train_path, duplicate_factor=6)

    model = get_model_for_finetune(ModelType.RESNET50, args.ways)
    results = train_and_test(model, classical_dataset, test_dataset, num_epochs=200, device=device)

    for (train_loss, train_acc, test_loss, test_acc) in zip(results.train_losses, results.train_accs, results.losses, results.accs):
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

    wandb.finish()