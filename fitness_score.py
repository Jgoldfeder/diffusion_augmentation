import os
import json
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256, FashionMNIST, ImageFolder, SUN397
from torch.utils.data import DataLoader
from CustomDataset import split_train_test, split_train_val
import torch
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch import nn
from make_augmenations_from_tree import generate_augmentations_from_tree
from AugmentationNode import initialize_augmentation_tree
from CustomDataset import TreeAugmentedDataset

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_datasets():
    root = './torch'

    # TODO make num_ways and num_shots come from args as well
    dataset_name = 'caltech256'
    num_ways = 5
    num_shots = 2

    if dataset_name == 'caltech256':
        dataset = Caltech256(root=root, download=True)
        class_to_label = dict()
        label_to_class = dict()
        for category in dataset.categories:
            parts = category.split('.')
            label = int(parts[0]) - 1
            class_name = parts[1]
            class_to_label[class_name] = label
            label_to_class[label] = class_name
        labels = [label for _, label in dataset]
    elif dataset_name == 'fashionmnist':
        dataset = FashionMNIST(root=root, download=True)
        class_to_label = dataset.class_to_idx
        label_to_class = {label: class_name for class_name, label in class_to_label.items()}
        labels = dataset.targets.tolist()
    elif dataset_name =='flowers102':
        data_dir = os.path.join(root, 'flower_data')
        try:
            dataset = ImageFolder(os.path.join(data_dir, 'train'))
        except FileNotFoundError:
            print('Download the dataset from kaggle from the following page using curl:')
            print('https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset')
        with open(os.path.join(data_dir, 'cat_to_name.json'), 'r') as f:
            label_to_class_as_str = json.load(f)
        label_to_class = {int(label_str): class_name for label_str, class_name in label_to_class_as_str.items()}
        class_to_label = {class_name: label for label, class_name in label_to_class.items()}
        labels = [int(target) for target in dataset.targets]
    elif dataset_name == 'sun397':
        dataset = SUN397(root=root, download=True)
        class_to_label = dataset.class_to_idx
        label_to_class = {label: class_name for class_name, label in class_to_label.items()}
        cache_file = os.path.join(root, 'cached_data', 'sun397_labels.json')
        try:
            with open(cache_file, 'r') as f:
                labels = json.load(f)
            print("Loaded labels from cache file.")
        except FileNotFoundError:
            print("This process may take ten minutes, but will also create cache file.")
            labels = []
            total = len(dataset)
            for i, (_, label) in enumerate(dataset):
                labels.append(label)
                if i % int(total * 0.1) == 0:
                    print(f"Completed {i/total*100:.1f}%")
            with open(cache_file, 'w') as f:
                json.dump(labels, f)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    base_dataset, test_dataset, old_to_new_labels = split_train_test(dataset, class_to_label, labels, num_ways, num_shots)

    new_label_to_class = [''] * len(old_to_new_labels)
    for old_label, new_label in old_to_new_labels.items():
        new_label_to_class[new_label] = label_to_class[old_label]

    

    train_dataset, val_dataset = split_train_val(base_dataset)

    return train_dataset, val_dataset, test_dataset, new_label_to_class

def fitness_score(individual) -> float:
    
    #first create the dataset
    train_dataset, val_dataset, test_dataset, label_to_class = create_datasets()
    val_dataset = TreeAugmentedDataset(val_dataset, label_to_class, transform)
    #split the base into train and val 50/50 but keep the same class distribution

    #then generate the augmentations on the train set
    augmented_dataset = generate_augmentations_from_tree(individual, train_dataset, label_to_class)
    augmented_dataset = TreeAugmentedDataset(augmented_dataset, label_to_class, transform)

    #then train the model on the train set
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 256)
    model.to("cuda")

    #train for 1 epoch and calculate validation loss after one epoch
    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        for batch in train_loader:
            print(batch)
            images = batch[0]
            labels = batch[1]
            
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'No shape'}")
            print(f"Labels content: {labels}")
            
            images = images.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        #calculate validation loss
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    return val_loss

#initialize some random tree and run the fitness score
individual = initialize_augmentation_tree(depth=4)
fitness_score(individual)
print(fitness_score(individual))