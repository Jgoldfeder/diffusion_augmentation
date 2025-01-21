import torch
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
from make_augmenations_from_tree import generate_augmentations_from_tree
from AugmentationNode import AugmentationNode
from AugmentationNode import initialize_augmentation_tree
import os
from CustomDataset import FewShotDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from torchvision import transforms

from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager


def visualize_augmentations(original_images, augmented_images):
    # Create output directory if it doesn't exist
    output_dir = "sample_tree_augmentations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original images
    for idx, img in enumerate(original_images):
        img.save(os.path.join(output_dir, f'original_{idx+1}.png'))
    
    # Save augmented images
    for idx, img in enumerate(augmented_images):
        img.save(os.path.join(output_dir, f'augmented_{idx+1}.png'))

def main():
    dataset_path = "few_shot_datasets/caltech256/2_shot/seed_41"
    train_dataset = FewShotDataset(dataset_path, dataset_type='train')
    test_dataset = FewShotDataset(dataset_path, dataset_type='test')

    segment_aug_manager = SegmentAugmentationManager()
    color_aug_manager = ColorControlNetAugmentationManager()
    canny_aug_manager = CannyAugmentationManager()
    nerf_aug_manager = NerfAugmentationManager()
    depth_aug_manager = DepthAugmentationManager()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug_tree = initialize_augmentation_tree(depth=4)
    augmented_dataset = generate_augmentations_from_tree(aug_tree, train_dataset, aug_managers, transform)

    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #train the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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


if __name__ == "__main__":
    main()