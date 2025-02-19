import pygad
import random
import numpy as np
import wandb
import argparse
import time
import os

import AugmentationNode
from AugmentationNode import print_tree
import fitness_score
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

import get_base_model
from make_augmenations_from_tree import generate_augmentations_from_tree
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch

from CustomDataset import FewShotDataset

segment_aug_manager = SegmentAugmentationManager()
color_aug_manager = ColorControlNetAugmentationManager()
canny_aug_manager = CannyAugmentationManager()
nerf_aug_manager = NerfAugmentationManager()
depth_aug_manager = DepthAugmentationManager()
aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]
augmentation_types = ['canny', 'depth', 'seg', 'color', 'nerf', 'classical', 'none']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_trees():
    trees = []
    for augmentation_type in augmentation_types:
        tree = AugmentationNode.AugmentationNode(augmentation_type=augmentation_type)
        trees.append(tree)
    return trees

def compute_test_accuracy(augmentation_tree, device, args):
    print('[LOG] Computing test accuracy on combined train and val datasets')
    print('For both classical augs and the best aug tree')

    dataset_path = f"few_shot_datasets/{args.dataset}/{args.num_shots}_shot/seed_{args.seed}"
    train_dataset = FewShotDataset(dataset_path, dataset_type='train')
    test_dataset = FewShotDataset(dataset_path, dataset_type='test')

    # classical_dataset = ClassicalDataset(train_dataset, transform, duplicate_factor=6)
    augmented_dataset = generate_augmentations_from_tree(augmentation_tree, train_dataset, aug_managers, transform)

    output_dir = 'sample_tree_augmentations'
    for idx, (img, label, class_name) in enumerate(augmented_dataset):
        # Generate the filename using the specified format
        file_name = f"{class_name}_{label}_{idx}.png"  # Use .png or desired image format
        file_path = os.path.join(output_dir, file_name)
        
        # Convert the tensor image to a PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # remove normalization
            img = transforms.ToPILImage()(img)  # Convert to PIL Image
        
        # Save the image to the specified folder
        img.save(file_path)


    model = get_base_model.get_resnet50(num_outputs=args.num_ways)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(400):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels, class_names in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels, class_names in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = epoch_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{400}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')

        wandb.log({
            f"train_loss_{augmentation_tree.augmentation_type}": avg_loss,
            f"train_accuracy_{augmentation_tree.augmentation_type}": train_accuracy,
            f"test_accuracy_{augmentation_tree.augmentation_type}": test_accuracy,
            "epoch": epoch
        })

    return test_accuracy

def main():
    parser = argparse.ArgumentParser(description='Run augmentation tree experiments')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['caltech256', 'flowers102'])
    parser.add_argument('--num_shots', type=int, required=True)
    parser.add_argument('--num_ways', type=int, required=True)
    args = parser.parse_args()
    
    wandb.init(
        project="depth1-tree-tests",
        config={
            "dataset": args.dataset,
            "seed": args.seed,
            "num_shots": args.num_shots,
            "num_ways": args.num_ways
        }
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trees = create_trees()
    for tree in trees:
        compute_test_accuracy(tree, device, args)

if __name__ == "__main__":
    main()