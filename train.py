import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256
from torchvision.models import resnet18, ResNet18_Weights
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

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, duplicate=1, use_diffusion_aug=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.duplicate = duplicate
        self.use_diffusion_aug = use_diffusion_aug
        
        if use_diffusion_aug:
            self.augmented_images = self._generate_diffusion_augmentations()
    
    def _generate_diffusion_augmentations(self):
        temp_paths = []
        for idx, img in enumerate(self.images):
            if hasattr(img, 'filename') and img.filename:
                temp_path = os.path.abspath(img.filename)
            else:
                temp_path = os.path.abspath(f'temp_img_{idx}.png')
                img.save(temp_path)
            temp_paths.append(temp_path)
        
        print(temp_paths)
        
        controlnet_manager = ControlNetAugmentationManager()
        canny_aug, depth_aug, seg_aug = controlnet_manager.generate_augmentations(temp_paths)
        
        color_manager = ColorControlNetAugmentationManager()
        color_aug = color_manager.generate_augmentations(temp_paths)
        
        nerf_manager = NerfAugmentationManager()
        nerf_aug = nerf_manager.generate_augmentations(temp_paths)
        
        augmented_images = {}
        for path in temp_paths:
            img_augs = []
            if path in canny_aug: img_augs.append(canny_aug[path])
            if path in depth_aug: img_augs.append(depth_aug[path])
            if path in seg_aug: img_augs.append(seg_aug[path])
            if path in color_aug: img_augs.append(color_aug[path])
            if path in nerf_aug: img_augs.append(nerf_aug[path])
            augmented_images[path] = img_augs
            
        return augmented_images
    
    def __len__(self):
        if self.use_diffusion_aug:
            return len(self.images) * 6
        return len(self.images) * self.duplicate
    
    def __getitem__(self, idx):
        if self.use_diffusion_aug:
            true_idx = idx // 6
            aug_idx = idx % 6
            
            if aug_idx == 0:
                image = self.images[true_idx]
                if self.transform:
                    image = self.transform(image)
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

def create_datasets():
    dataset = Caltech256(root='./torch', download=True)
    
    all_classes = list(set([label for _, label in dataset]))
    selected_classes = random.sample(all_classes, 5)
    
    class_images = {c: [] for c in selected_classes}
    for img, label in dataset:
        if label in selected_classes:
            img = img.convert('RGB')
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
                valid_images.append(img)
        
        if len(valid_images) < 2:
            raise ValueError(f"Not enough valid images found for class {class_idx}. Need at least 2, found {len(valid_images)}")
        
        selected_imgs = random.sample(valid_images, 2)
        train_images.extend(selected_imgs)
        train_labels.extend([label] * 2)
        
        remaining_imgs = [img for img in class_images[class_idx] if img not in selected_imgs]
        test_images.extend(remaining_imgs)
        test_labels.extend([label] * len(remaining_imgs))

        label += 1
    
    augmented_dataset = CustomDataset(train_images, train_labels, basic_transform, use_diffusion_aug=True)
    original_dataset = CustomDataset(train_images, train_labels, basic_transform, duplicate=6)
    test_dataset = CustomDataset(test_images, test_labels, basic_transform)
    
    return augmented_dataset, original_dataset, test_dataset

def train_model(train_dataset, test_dataset, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
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
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Evaluation loop
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
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = epoch_loss / len(train_loader)
        
        # Log metrics to wandb
        wandb.log({
            f"{model_name}/train_loss": avg_loss,
            f"{model_name}/train_accuracy": train_accuracy,
            f"{model_name}/test_accuracy": test_accuracy,
            "epoch": epoch
        })
        
        print(f'{model_name} - Epoch {epoch+1}/{epochs}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

def main():
    augmented_dataset, original_dataset, test_dataset = create_datasets()
    
    aug_losses, aug_accuracies = train_model(augmented_dataset, test_dataset, "Augmented")
    orig_losses, orig_accuracies = train_model(original_dataset, test_dataset, "Original")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(aug_losses, label='Augmented')
    plt.plot(orig_losses, label='Original')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(aug_accuracies, label='Augmented')
    plt.plot(orig_accuracies, label='Original')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    main()
