import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

class DiffAugImage:
	def __init__(self, image, label, is_augmented):
		self.image = image
		self.label = label
		self.is_augmented = is_augmented

class ImageDataset(Dataset):
    def __init__(self, diff_aug_images: List[DiffAugImage], transform=None):
        """
        Args:
            images (list): List of PIL Image objects.
            labels (list): List of class labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.diff_aug_images: List[DiffAugImage] = diff_aug_images
        self.transform = transform

    def __len__(self):
        return len(self.diff_aug_images)

    def __getitem__(self, idx):
        image = self.diff_aug_images[idx].image
        label = self.diff_aug_images[idx].label

        if self.transform:
            image = self.transform(image)
        # label = torch.tensor(int(label), dtype=torch.long)

        return image, label

def finetune_resnet(base_dataset, test_dataset, num_classes, epochs=10, batch_size=1, learning_rate=1e-3):
    """
    Fine-tunes a ResNet model on the given base dataset and evaluates accuracy on the test dataset.

    Args:
        base_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
        num_classes (int): Number of classes in the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        None
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet pre-trained on ImageNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace the fully connected layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Move model to device (GPU/CPU)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoaders
    train_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	class_names = ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove']

	base_images_per_class = 2
	augmented_images_per_class = 8
	test_images_per_class = 90

	base_images = []
	for class_name in class_names:
		input_directory = os.path.join('torch/caltech256/256_ObjectCategories', class_name)
		for i in range(base_images_per_class):
			img_id = i + 1
			class_label = class_name.split('.')[0]
			image_name = class_label + "_" + str(img_id).zfill(4) + ".jpg"
			img = Image.open(os.path.join(input_directory, image_name)).convert('RGB')
			base_images.append(DiffAugImage(img, int(class_label) - 1, False))

	augmented_images = []
	# TODO actually make this get augmented images
	# for now just assumes the "augmented images" are just normal ones not in the base images list
	for class_name in class_names:
		input_directory = os.path.join('torch/caltech256/256_ObjectCategories', class_name)
		for i in range(3, augmented_images_per_class + 3):
			class_label = class_name.split('.')[0]
			image_name = class_label + "_" + str(i).zfill(4) + ".jpg"
			img = Image.open(os.path.join(input_directory, image_name)).convert('RGB')
			augmented_images.append(DiffAugImage(img, int(class_label) - 1, True))

	repeated_images = []
	while len(repeated_images) < len(augmented_images) + len(base_images):
		repeated_images.append(base_images[len(repeated_images) % len(base_images)])

	test_images = []
	for class_name in class_names:
		input_directory = os.path.join('torch/caltech256/256_ObjectCategories', class_name)
		for i in range(test_images_per_class):
			img_id = i + 1
			class_label = class_name.split('.')[0]
			image_name = class_label + "_" + str(img_id).zfill(4) + ".jpg"
			img = Image.open(os.path.join(input_directory, image_name)).convert('RGB')
			test_images.append(DiffAugImage(img, int(class_label) - 1, False))


	base_dataset = ImageDataset(base_images, transform=transform)
	repeated_dataset = ImageDataset(repeated_images, transform=transform)
	augmented_dataset = ImageDataset(base_images + augmented_images, transform=transform)
	test_dataset = ImageDataset(test_images, transform=transform)

	print(len(base_dataset))
	print(len(repeated_dataset))
	print(len(augmented_dataset))

	num_classes = len(class_names)

	print('base')
	finetune_resnet(base_dataset, test_dataset, num_classes, epochs=20, batch_size=1, learning_rate=1e-4)
	print('repeated')
	finetune_resnet(repeated_dataset, test_dataset, num_classes, epochs=20, batch_size=1, learning_rate=1e-4)
	print('augmented')
	finetune_resnet(augmented_dataset, test_dataset, num_classes, epochs=20, batch_size=1, learning_rate=1e-4)
