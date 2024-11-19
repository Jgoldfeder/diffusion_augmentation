# We want:

# 1. A dataset which will include augmented images along with normal images
# 2. A dataset which will include just normal images, with a repitition number
#         so if rep number is 3, it includes each normal image replicated 2 more times, for a total of 3 of the 
#         same image in the dataset
# 3. A test dataset which will include all the normal images from our dataset

# We should be able to pass in the name of the dataset, ex. 'torch/caltech256'

# We should also be able to pass in the names of the classes we are interested in
# ex. ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove']

# We should be able to pass in for the augmented images a list of paths to types of augmentations
# for example ['Canny', 'color_controlnet', 'Depth', 'Segmentation', 'zero123']
# and it should create the dataset using the images from each type of augmentation

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import List, Optional, Dict, Tuple
import copy
from collections import defaultdict
import random

from torchvision.transforms.functional import to_pil_image


class BaseImageDataset:
    """Base class to load and manage the original dataset."""
    
    def __init__(self, dataset_name: str, selected_classes: List[str], downloaded_dataset_path: str):
        self.dataset_name = dataset_name
        self.selected_classes = selected_classes
        self.downloaded_dataset_path = downloaded_dataset_path
        self.class_to_idx = {}
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the original dataset and filter by selected classes."""
        if self.dataset_name == 'caltech256':
            # Create class_to_idx mapping for selected classes
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.selected_classes)
            }
            
            # Assuming the Caltech256 structure: root/256_ObjectCategories/class_name/image.jpg
            categories_dir = os.path.join(self.downloaded_dataset_path, '256_ObjectCategories')
            
            # Filter samples based on selected classes
            for class_name in self.selected_classes:
                class_dir = os.path.join(categories_dir, class_name)
                if not os.path.exists(class_dir):
                    raise ValueError(f"Class directory not found: {class_dir}")
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((
                            img_path,
                            self.class_to_idx[class_name]
                        ))
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

class AugmentedDataset(Dataset):
    """Dataset that includes only normal images with augmentations and their corresponding augmented images."""
    
    def __init__(
        self,
        base_dataset: BaseImageDataset,
        augmentation_paths: List[str],
        transform: Optional[transforms.Compose] = None
    ):
        self.base_dataset = base_dataset
        self.augmentation_paths = augmentation_paths
        self.transform = transform
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Tuple[str, int]]:
        """Prepare list of samples including only normal images with augmentations and their augmentations."""
        valid_samples = []
        augmented_samples = []
        
        # Check if augmented images exist for each original image
        for orig_path, class_idx in self.base_dataset.samples:
            img_name = os.path.basename(orig_path)
            class_name = os.path.basename(os.path.dirname(orig_path))
            
            # Check if augmentations exist for this image
            has_augmentations = False
            for aug_path in self.augmentation_paths:
                aug_class_path = os.path.join(aug_path, class_name)
                aug_image_path = os.path.join(aug_class_path, img_name)
                if os.path.exists(aug_image_path):
                    augmented_samples.append((aug_image_path, class_idx))
                    has_augmentations = True
            
            # Include original image only if augmentations exist
            if has_augmentations:
                valid_samples.append((orig_path, class_idx))
        
        # Combine valid original images and their augmentations
        all_samples = valid_samples + augmented_samples
        return all_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx


class RepeatedDataset(Dataset):
    """Dataset that includes repeated original images."""
    
    def __init__(
        self,
        base_dataset: BaseImageDataset,
        repetitions: int,
        images_per_class: Optional[int] = None,
        seed: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the RepeatedDataset.
        
        Args:
            base_dataset: The base dataset containing original samples
            repetitions: Number of times to repeat each selected image
            images_per_class: Maximum number of images to select per class before repetition
            seed: Random seed for reproducible image selection
            transform: Optional transforms to apply to the images
        """
        self.base_dataset = base_dataset
        self.repetitions = repetitions
        self.images_per_class = images_per_class
        self.transform = transform
        
        if seed is not None:
            random.seed(seed)
            
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> List[Tuple[str, int]]:
        """Prepare list of samples with repetitions."""
        # Group samples by class
        samples_by_class = defaultdict(list)
        for sample_path, class_idx in self.base_dataset.samples:
            samples_by_class[class_idx].append((sample_path, class_idx))
        
        # Select and repeat samples
        final_samples = []
        for class_idx, class_samples in samples_by_class.items():
            # If images_per_class is specified and less than available samples,
            # randomly select that many images
            if self.images_per_class is not None and self.images_per_class < len(class_samples):
                selected_samples = random.sample(class_samples, self.images_per_class)
            else:
                selected_samples = class_samples
            
            # Add the selected samples repeated times
            final_samples.extend(selected_samples * self.repetitions)
        
        return final_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

class TestDataset(Dataset):
    """Dataset for testing that includes only original images."""
    
    def __init__(
        self,
        base_dataset: BaseImageDataset,
        transform: Optional[transforms.Compose] = None
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.samples = self.base_dataset.samples
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

def create_datasets(
    dataset_name: str,
    selected_classes: List[str],
    downloaded_dataset_path: str,
    augmentation_paths: List[str],
    repetitions: int,
    images_per_class: Optional[int] = None,
    seed: Optional[int] = None,
    transform: Optional[transforms.Compose] = None
) -> Tuple[AugmentedDataset, RepeatedDataset, TestDataset]:
    """
    Create all three datasets with the specified parameters.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'caltech256')
        selected_classes: List of class names to include
        downloaded_dataset_path: Path to the downloaded dataset
        augmentation_paths: List of paths to augmented images
        repetitions: Number of times to repeat each image in RepeatedDataset
        images_per_class: Maximum number of images to select per class in RepeatedDataset
        seed: Random seed for reproducible image selection
        transform: Optional transforms to apply to the images
        
    Returns:
        Tuple of (AugmentedDataset, RepeatedDataset, TestDataset)
    """
    base_dataset = BaseImageDataset(dataset_name, selected_classes, downloaded_dataset_path)
    
    augmented_dataset = AugmentedDataset(
        base_dataset,
        augmentation_paths,
        transform
    )
    
    repeated_dataset = RepeatedDataset(
        base_dataset,
        repetitions,
        images_per_class,
        seed,
        transform
    )
    
    test_dataset = TestDataset(
        base_dataset,
        transform
    )
    
    return augmented_dataset, repeated_dataset, test_dataset


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report

# Function to fine-tune a ResNet model
def fine_tune_resnet(train_dataset, test_dataset, num_classes, epochs=5, batch_size=32, lr=0.001, device="cuda"):
    """
    Fine-tune a ResNet model on a training dataset and evaluate on a test dataset.

    Args:
        train_dataset: Dataset used for training.
        test_dataset: Dataset used for testing.
        num_classes: Number of classes in the dataset.
        epochs: Number of training epochs.
        batch_size: Batch size for training and testing.
        lr: Learning rate for the optimizer.
        device: Device to use for training ('cuda' or 'cpu').

    Returns:
        model: The fine-tuned model.
        test_metrics: Metrics for the test dataset evaluation.
    """
    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained ResNet and modify the final layer
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate on the test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification report
    test_metrics = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    
    return model, test_metrics


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    selected_classes = [
        '001.ak47',
        '002.american-flag',
        '003.backpack',
        '004.baseball-bat',
        '005.baseball-glove'
    ]

    # Paths configuration
    downloaded_dataset_path = 'torch/caltech256'  # Replace with actual path
    augmentation_path_base = 'augmented_images/caltech256'
    augmentation_paths = [
        f'{augmentation_path_base}/Canny',
        f'{augmentation_path_base}/color_controlnet',
        f'{augmentation_path_base}/Depth',
        f'{augmentation_path_base}/Segmentation',
        f'{augmentation_path_base}/zero123'
    ]

    # Create datasets with limit of 10 images per class in repeated dataset
    augmented_dataset, repeated_dataset, test_dataset = create_datasets(
        dataset_name='caltech256',
        selected_classes=selected_classes,
        downloaded_dataset_path=downloaded_dataset_path,
        augmentation_paths=augmentation_paths,
        repetitions=6,
        images_per_class=2,
        seed=42,  # For reproducible results
        transform=transform
    )

    print(len(augmented_dataset))
    print(len(repeated_dataset))
    print(len(test_dataset))


    # Number of classes
    num_classes = len(selected_classes)

    # Fine-tune on the RepeatedDataset
    print("Training on RepeatedDataset")
    repeated_model, repeated_metrics = fine_tune_resnet(
        train_dataset=repeated_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        epochs=100,
        batch_size=1,
        lr=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Fine-tune on the AugmentedDataset
    print("Training on AugmentedDataset")
    augmented_model, augmented_metrics = fine_tune_resnet(
        train_dataset=augmented_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        epochs=100,
        batch_size=1,
        lr=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Compare metrics
    import pandas as pd

    repeated_results = pd.DataFrame(repeated_metrics).T
    augmented_results = pd.DataFrame(augmented_metrics).T

    print("\nPerformance Comparison:")
    print("Repeated Dataset Results:")
    print(repeated_results)

    print("\nAugmented Dataset Results:")
    print(augmented_results)


