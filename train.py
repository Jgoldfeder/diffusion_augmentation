import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256, SUN397
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import os
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
import wandb
import argparse
from CustomDataset import create_datasets

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)
# set a random seed

basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
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
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--dataset', type=str, choices=['caltech256', 'sun397'], 
                       required=True, help='Dataset to use (caltech256 or sun397)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def get_model(architecture, num_classes):
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True

    return model

def create_datasets_backup(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'caltech256':
        dataset = Caltech256(root='./torch', download=True)
        all_classes = []
        for _, label in dataset:
            all_classes.append(label)
    else:
        dataset = SUN397(root='./torch', download=True) 
        all_classes = dataset.classes
    num_classes = 397 if args.dataset == 'sun397' else 256
    selected_classes = random.sample(all_classes, 5)
    print(f"<LOG> Selected classes: {selected_classes}")
    
    class_images = {c: [] for c in selected_classes}
    
    if args.dataset == 'sun397':
        for class_idx in selected_classes:
            print(f"<LOG> Processing class {class_idx}")
            first_letter = class_idx[0]
            data_path = os.path.join(os.getcwd(), dataset.root, "SUN397", first_letter, class_idx)
            print(f"<LOG> Data path: {data_path}")
            for img_name in os.listdir(data_path):
                img_path = os.path.join(data_path, img_name)
                img = Image.open(img_path).convert("RGB").resize((512, 512))
                img.filename = img_path
                class_images[class_idx].append(img)
    elif args.dataset == 'caltech256':
        data_path = os.path.join(os.getcwd(), dataset.root, "caltech256", "256_ObjectCategories")
        for img, label in dataset:
            if label in selected_classes:
                img_path = os.path.join(data_path, img.filename)
                img = img.convert("RGB")
                img.filename = img_path
                class_images[label].append(img)
            
    print("<LOG> Finished creating class images")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    label = 0
    for class_idx in selected_classes:
        valid_images = []
        for img in class_images[class_idx]:
            if hasattr(img, 'filename') and img.filename and os.path.exists(img.filename):
                old_filename = img.filename
                #img = img.convert('RGB').resize((512,`512))
                img.filename = old_filename
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
    
    model = get_model(args.architecture, num_classes=5)
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
            #print("HERE IN TESTING")
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
        project="diffusion-augmentation",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "architecture": args.architecture,
            "dataset": args.dataset,
            "use_canny": args.use_canny,
            "use_depth": args.use_depth,
            "use_seg": args.use_seg,
            "use_color": args.use_color,
            "use_nerf": args.use_nerf,
            "seed": args.seed
            #"num_classes": args.num_classes,
            #"images_per_class": args.images_per_class
            #"use_nerf": args.use_nerf
        }
    )
    
    augmented_dataset, original_dataset, test_dataset = create_datasets(args)
    
    #train_model(augmented_dataset, test_dataset, "Augmented", args)
    train_model(original_dataset, test_dataset, "Original", args)
    
    wandb.finish()

if __name__ == "__main__":
    main()
