import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import os
from augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager
from augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from augmentation_models.NerfAugmentation import NerfAugmentationManager
import wandb
import argparse

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 with various augmentations')
    parser.add_argument('--use_canny', action='store_true', help='Use ControlNet Canny augmentation')
    parser.add_argument('--use_depth', action='store_true', help='Use ControlNet Depth augmentation')
    parser.add_argument('--use_seg', action='store_true', help='Use ControlNet Segmentation augmentation')
    parser.add_argument('--use_color', action='store_true', help='Use Color ControlNet augmentation')
    parser.add_argument('--use_nerf', action='store_true', help='Use NeRF augmentation')
    
    parser.add_argument('--architecture', type=str, default='resnet18', help='Model architecture (e.g., resnet18, resnet50)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    
    return parser.parse_args()

def get_model(architecture, num_classes):
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, duplicate=1, use_diffusion_aug=False, args=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.args = args
        self.use_diffusion_aug = use_diffusion_aug
        
        if use_diffusion_aug:
            self.augmented_images = self._generate_diffusion_augmentations()
            self.num_augmentations = sum([
                self.args.use_canny,
                self.args.use_depth,
                self.args.use_seg,
                self.args.use_color,
                self.args.use_nerf
            ])
            self.duplicate = self.num_augmentations + 1
        else:
            self.duplicate = duplicate
    
    def _generate_diffusion_augmentations(self):
        temp_paths = []
        for idx, img in enumerate(self.images):
            if hasattr(img, 'filename') and img.filename:
                temp_path = os.path.abspath(img.filename)
            else:
                temp_path = os.path.abspath(f'temp_img_{idx}.png')
                img.save(temp_path)
            temp_paths.append(temp_path)
        
        if self.args.use_canny or self.args.use_depth or self.args.use_seg:
            controlnet_manager = ControlNetAugmentationManager()
            canny_aug, depth_aug, seg_aug = controlnet_manager.generate_augmentations(temp_paths)
        if self.args.use_color:
            color_manager = ColorControlNetAugmentationManager()
            color_aug = color_manager.generate_augmentations(temp_paths)
        if self.args.use_nerf:
            nerf_manager = NerfAugmentationManager()
            nerf_aug = nerf_manager.generate_augmentations(temp_paths)
                
        augmented_images = {}
        for path in temp_paths:
            img_augs = []
            if self.args.use_canny and path in canny_aug:
                img_augs.append(canny_aug[path])
            if self.args.use_depth and path in depth_aug:
                img_augs.append(depth_aug[path])
            if self.args.use_seg and path in seg_aug:
                img_augs.append(seg_aug[path])
            if self.args.use_color and path in color_aug:
                img_augs.append(color_aug[path])
            if self.args.use_nerf and path in nerf_aug:
                img_augs.append(nerf_aug[path])
            augmented_images[path] = img_augs
                
        return augmented_images
    
    def __len__(self):
        return len(self.images) * self.duplicate
    
    def __getitem__(self, idx):
        if self.use_diffusion_aug:
            true_idx = idx // self.duplicate
            aug_idx = idx % self.duplicate
            
            if aug_idx == 0:
                image = self.images[true_idx]
            else:
                if hasattr(self.images[true_idx], 'filename') and self.images[true_idx].filename:
                    img_path = os.path.abspath(self.images[true_idx].filename)
                else:
                    img_path = os.path.abspath(f'temp_img_{true_idx}.png')
                
                image = self.augmented_images[img_path][aug_idx - 1]
            
            if self.transform:
                image = self.transform(image)
            return image, self.labels[true_idx]
        else:
            true_idx = idx // self.duplicate
            image = self.images[true_idx]
            if self.transform:
                image = self.transform(image)
            return image, self.labels[true_idx]

def create_datasets(args):
    dataset = Caltech256(root='./torch', download=True)
    
    all_classes = list(set([label for _, label in dataset]))
    selected_classes = random.sample(all_classes, 5)
    
    class_images = {c: [] for c in selected_classes}
    for img, label in dataset:
        if label in selected_classes:
            class_images[label].append(img)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    label = 0
    for class_idx in selected_classes:
        # Filter for images that exist in the directory
        valid_images = []
        for img in class_images[class_idx]:
            if hasattr(img, 'filename') and img.filename and os.path.exists(img.filename):
                old_filename = img.filename
                img = img.convert('RGB')
                img.filename = old_filename
                valid_images.append(img)
        
        if len(valid_images) < 2:
            raise ValueError(f"Not enough valid images found for class {class_idx}. Need at least 2, found {len(valid_images)}")
        
        selected_imgs = random.sample(valid_images, 2)
        train_images.extend(selected_imgs)
        train_labels.extend([label] * 2)
        
        remaining_imgs = [img.convert('RGB') for img in class_images[class_idx] if img not in selected_imgs]
        test_images.extend(remaining_imgs)
        test_labels.extend([label] * len(remaining_imgs))

        label += 1
    
    augmented_dataset = CustomDataset(train_images, train_labels, basic_transform, use_diffusion_aug=True, args=args)
    original_dataset = CustomDataset(train_images, train_labels, basic_transform, 
                                   duplicate=augmented_dataset.duplicate)
    test_dataset = CustomDataset(test_images, test_labels, basic_transform)
    
    print(f"Dataset sizes:")
    print(f"  Augmented training set: {len(augmented_dataset)} images")
    print(f"  Original training set: {len(original_dataset)} images") 
    print(f"  Test set: {len(test_dataset)} images")
    return augmented_dataset, original_dataset, test_dataset

def train_model(train_dataset, test_dataset, dataset_type, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
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
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = epoch_loss / len(train_loader)
        
        wandb.log({
            f"{dataset_type}/train_loss": avg_loss,
            f"{dataset_type}/train_accuracy": train_accuracy,
            f"{dataset_type}/test_accuracy": test_accuracy,
            "epoch": epoch
        })
        
        print(f'{dataset_type} - Epoch {epoch+1}/{args.epochs}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')
    
    return avg_loss, train_accuracy, test_accuracy

def main():
    args = parse_args()
    
    wandb.init(
        project="caltech256-augmentation",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "architecture": args.architecture,
            "dataset": "Caltech256",
            "use_canny": args.use_canny,
            "use_depth": args.use_depth,
            "use_seg": args.use_seg,
            "use_color": args.use_color,
            "use_nerf": args.use_nerf
        }
    )
    
    augmented_dataset, original_dataset, test_dataset = create_datasets(args)
    
    train_model(augmented_dataset, test_dataset, "Augmented", args)
    train_model(original_dataset, test_dataset, "Original", args)
    
    wandb.finish()

if __name__ == "__main__":
    main()
