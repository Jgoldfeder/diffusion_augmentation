import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


class FewShotDatasetAugmented(Dataset):
    def __init__(self, root_dir, dataset, categories, transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.categories = categories
        self.transform = transform
        self.data_paths, self.labels = self._create_data_paths()

    def _create_data_paths(self):
        data_paths = []
        labels = []
        
        # Loop through each category, assigning a label for each
        for label, category in enumerate(self.categories):
            # Add normal images as individual entries
            normal_images = glob.glob(os.path.join(self.root_dir, "normal_images", self.dataset, category, "*.jpg"))
            for normal_img_path in normal_images:
                data_paths.append(normal_img_path)
                labels.append(label)
                
                # Add each augmentation as an individual entry
                img_id = normal_img_path.split("/")[-1].split(".")[0]
                for aug_type in ["Canny", "color_controlnet", "Depth", "Segmentation", "zero123"]:
                    augmented_img_path = os.path.join(self.root_dir, "augmented_images", self.dataset, aug_type, category, img_id + ".jpg")
                    data_paths.append(augmented_img_path)
                    labels.append(label)
                    
        return data_paths, labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Retrieve the image path and label
        img_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

class FewShotDatasetNormal(Dataset):
    def __init__(self, root_dir, dataset, categories, transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.categories = categories
        self.transform = transform
        self.data_paths, self.labels = self._create_data_paths_and_labels()

    def _create_data_paths_and_labels(self):
        data_paths = []
        labels = []
        # Loop through each category, gathering paths and labels
        for label, category in enumerate(self.categories):
            normal_images = glob.glob(os.path.join(self.root_dir, "normal_images", self.dataset, category, "*.jpg"))
            data_paths.extend(normal_images)
            labels.extend([label] * len(normal_images))  # Assign label for each image in the category
        return data_paths, labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        normal_img_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load the normal image
        normal_image = Image.open(normal_img_path).convert("RGB")
        if self.transform:
            normal_image = self.transform(normal_image)

        return normal_image, label


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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

categories = ["001.ak47", "002.american-flag", "003.backpack", "004.baseball-bat", "005-baseball-glove"]

dataset_normal = FewShotDatasetNormal(root_dir=".", dataset="caltech256", categories=categories, transform=transform)
dataset_augmented = FewShotDatasetAugmented(root_dir=".", dataset="caltech256", categories=categories, transform=transform)

train_loader_normal = DataLoader(dataset_normal, batch_size=1, shuffle=True)
train_loader_augmented = DataLoader(dataset_augmented, batch_size=1, shuffle=True)

model_normal = models.resnet18(pretrained=True)
model_augmented = models.resnet18(pretrained=True)

num_ftrs = model_normal.fc.in_features
model_normal.fc = nn.Linear(num_ftrs, len(categories))  # Output layer matches number of categories
model_augmented.fc = nn.Linear(num_ftrs, len(categories))  # Output layer matches number of categories


model_normal = model_normal.to('cuda:0' if torch.cuda.is_available() else 'cpu')
model_augmented = model_augmented.to('cuda:0' if torch.cuda.is_available() else 'cpu')


optimizer_normal = optim.Adam(model_normal.parameters(), lr=1e-4)
optimizer_augmented = optim.Adam(model_augmented.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# #normal training loop
# for epoch in range(600):
#     model_augmented.train()
#     running_loss = 0.0
#     for images, labels in train_loader_normal:
#         images = images.to('cuda:0' if torch.cuda.is_available() else 'cpu')
#         labels = labels.to('cuda:0' if torch.cuda.is_available() else 'cpu')

#         optimizer_normal.zero_grad()
        
#         outputs = model_normal(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer_normal.step()
        
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader_normal)}")


# # Test dataset and DataLoader
# test_dataset = TestDataset(root_dir=".", dataset="caltech256", categories=categories, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Testing loop
# model_normal.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to('cuda:0' if torch.cuda.is_available() else 'cpu'), labels.to('cuda:0' if torch.cuda.is_available() else 'cpu')
#         outputs = model_normal(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total}%")


#augmented training loop
for epoch in range(100):
    model_augmented.train()
    running_loss = 0.0
    for images, labels in train_loader_augmented:
        # Move data to the appropriate device
        images = images.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Zero the parameter gradients
        optimizer_augmented.zero_grad()

        # Forward pass
        outputs = model_augmented(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer_augmented.step()
        
        # Accumulate the loss
        running_loss += loss.item()
    
    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader_augmented)}")


# Test dataset and DataLoader
test_dataset = TestDataset(root_dir=".", dataset="caltech256", categories=categories, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testing loop
model_augmented.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cuda:0' if torch.cuda.is_available() else 'cpu'), labels.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        outputs = model_augmented(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")