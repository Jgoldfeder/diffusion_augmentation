import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models


# Dataset Definitions
class FewShotDataset(Dataset):
    def __init__(self, root_dir, dataset, categories, transform=None, augmented=False):
        self.root_dir = root_dir
        self.dataset = dataset
        self.categories = categories
        self.transform = transform
        self.augmented = augmented
        self.data_paths, self.labels = self._create_data_paths()

    def _create_data_paths(self):
        data_paths = []
        labels = []
        for label, category in enumerate(self.categories):
            normal_images = glob.glob(os.path.join(self.root_dir, "normal_images", self.dataset, category, "*.jpg"))
            for normal_img_path in normal_images:
                data_paths.append(normal_img_path)
                labels.append(label)
                if self.augmented:
                    img_id = os.path.splitext(os.path.basename(normal_img_path))[0]
                    for aug_type in ["Canny", "color_controlnet", "Depth", "Segmentation", "zero123"]:
                        augmented_img_path = os.path.join(
                            self.root_dir, "augmented_images", self.dataset, aug_type, category, f"{img_id}.jpg"
                        )
                        data_paths.append(augmented_img_path)
                        labels.append(label)
        return data_paths, labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, dataset, categories, transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.categories = categories
        self.transform = transform
        self.image_paths, self.labels = self._create_test_data()

    def _create_test_data(self):
        image_paths = []
        labels = []
        for idx, category in enumerate(self.categories):
            category_images = glob.glob(os.path.join(self.root_dir, "torch", self.dataset, "256_ObjectCategories", category, "*.jpg"))
            image_paths.extend(category_images)
            labels.extend([idx] * len(category_images))
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Helper Functions
def create_dataloader(dataset, batch_size=32, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def initialize_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    return accuracy


# Configuration
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

categories = ["001.ak47", "002.american-flag", "003.backpack", "004.baseball-bat", "005-baseball-glove"]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Dataset & DataLoader
dataset_normal = FewShotDataset(root_dir=".", dataset="caltech256", categories=categories, transform=transform)
dataset_augmented = FewShotDataset(root_dir=".", dataset="caltech256", categories=categories, transform=transform, augmented=True)

train_loader_normal = create_dataloader(dataset_normal, batch_size=32, shuffle=True)
train_loader_augmented = create_dataloader(dataset_augmented, batch_size=32, shuffle=True)

test_dataset = TestDataset(root_dir=".", dataset="caltech256", categories=categories, transform=transform)
test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

# Normal Model Training & Evaluation
model_normal = initialize_model(num_classes=len(categories), device=device)
optimizer_normal = optim.Adam(model_normal.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

train_model(model_normal, train_loader_normal, criterion, optimizer_normal, device, num_epochs=600)
evaluate_model(model_normal, test_loader, device)

# Augmented Model Training & Evaluation
model_augmented = initialize_model(num_classes=len(categories), device=device)
optimizer_augmented = optim.Adam(model_augmented.parameters(), lr=1e-4)

train_model(model_augmented, train_loader_augmented, criterion, optimizer_augmented, device, num_epochs=100)
evaluate_model(model_augmented, test_loader, device)
