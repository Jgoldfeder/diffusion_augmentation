import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256, SUN397, Flowers102, FashionMNIST, ImageFolder
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
import random
import json
from collections import defaultdict
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager


def get_label_remapping(old_labels_set):
    old_to_new_labels = dict()
    old_labels_sorted = sorted(list(set(old_labels_set)))
    for i in range(len(old_labels_sorted)):
        old_to_new_labels[old_labels_sorted[i]] = i
    return old_to_new_labels

# use this class so that labels start from 0 and go up, and
# thus have their original label remapped
class RemappedDataset(Dataset):
    def __init__(self, dataset, old_to_new_labels):
        self.dataset = dataset
        self.old_to_new_labels = old_to_new_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.convert('RGB'), self.old_to_new_labels[label]

def split_train_test(dataset, class_to_label, labels, num_ways, num_shots):
    target_classes = random.sample(class_to_label.keys(), num_ways)
    print("[LOG] Selected classes: ", target_classes)
    target_labels = [class_to_label[target_class] for target_class in target_classes]

    label_to_indexes = defaultdict(list)
    for index, label in enumerate(labels):
        if label in target_labels:
            label_to_indexes[label].append(index)

    train_indexes = []
    test_indexes = []
    for label, indexes in label_to_indexes.items():
        random.shuffle(indexes)
        train_indexes.extend(indexes[:num_shots])
        test_indexes.extend(indexes[num_shots:])

    old_to_new_labels = get_label_remapping(set(target_labels))
    train_dataset = RemappedDataset(Subset(dataset, train_indexes), old_to_new_labels)
    test_dataset = RemappedDataset(Subset(dataset, test_indexes), old_to_new_labels)

    print(len(train_dataset), len(test_dataset))

    return train_dataset, test_dataset, old_to_new_labels

def split_train_val(dataset, split_ratio=0.5):
    """
    Split a dataset into train and validation sets while maintaining class distribution
    Args:
        dataset: RemappedDataset object
        split_ratio: Proportion of data to use for training (default 0.5 for 50/50 split)
    """
    # Get all labels
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # Split each class according to ratio
    for label in label_to_indices:
        indices = label_to_indices[label]
        random.shuffle(indices)
        split_point = int(len(indices) * split_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    old_to_new_labels = get_label_remapping(set(labels))
    train_dataset = RemappedDataset(Subset(dataset, train_indices), old_to_new_labels)
    val_dataset = RemappedDataset(Subset(dataset, val_indices), old_to_new_labels)

    #assert that the train and val sets have the same class distribution
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
    assert len(set(train_labels)) == len(set(val_labels)), "Train and validation sets have different class distributions"

    return train_dataset, val_dataset

class ValDataset(Dataset):
    def __init__(self, orig_dataset, transform):
        self.dataset = orig_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label, class_name = self.dataset[index]
        return self.transform(img), label


class SplitDataset(Dataset):
    def __init__(self, dataset_list):
        self.images = []
        self.labels = []
        self.class_names = []
        for img, label, class_name in dataset_list:
            self.images.append(img)
            self.labels.append(label)
            self.class_names.append(class_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.class_names[index]

def split_into_two(dataset):
    images_0 = []
    labels_0 = []
    class_names_0 = []
    images_1 = []
    labels_1 = []
    class_names_1 = []

    for img, label, class_name in dataset:
        if labels_0.count(label) < labels_1.count(label):
            images_0.append(img)
            labels_0.append(label)
            class_names_0.append(class_name)
        else:
            images_1.append(img)
            labels_1.append(label)
            class_names_1.append(class_name)

    split_0 = [(images_0[i], labels_0[i], class_names_0[i]) for i in range(len(labels_0))]
    split_1 = [(images_1[i], labels_1[i], class_names_1[i]) for i in range(len(labels_1))]
    return SplitDataset(split_0), SplitDataset(split_1)


class FewShotDataset(Dataset): #dataset containing images from predefined structure
    def __init__(self, file_path, dataset_type="train"):
        self.images = []
        self.labels = []
        self.class_names = []
        self.dataset_type = dataset_type
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for class_name in os.listdir(os.path.join(file_path, self.dataset_type)):
            for img in os.listdir(os.path.join(file_path, self.dataset_type, class_name)):
                if img.endswith('.png'):
                    img = Image.open(os.path.join(file_path, self.dataset_type, class_name, img))
                    self.images.append(img)
                    self.labels.append(int(class_name.split('_')[0]))
                    self.class_names.append(class_name.split('_')[1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        if self.dataset_type == "test":
            img = self.transform(img.convert('RGB'))
        label = self.labels[index]
        class_names = self.class_names[index]
        return img, label, class_names

# TODO put the classical transform in the get function rather than creating the list at initialization
class ClassicalDataset(Dataset):
    def __init__(self, dataset, basic_transform, duplicate_factor=1):
        classical_aug_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop(size=(224, 224)),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2 
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomRotation(degrees=10)
        ])
        
        self.basic_transform = basic_transform
        self.dataset = []

        for img, label, class_name in dataset:
            self.dataset.append((self.basic_transform(img), label, class_name))
            for _ in range(duplicate_factor-1):
                augmented_img = classical_aug_transform(img)
                self.dataset.append((self.basic_transform(augmented_img), label, class_name))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label, class_name = self.dataset[index]
        return img, label, class_name


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, label_to_class, transform, args):
        self.transform = transform
        self.args = args

        base_images = []
        self.labels = []
        classes = []

        for img, label in base_dataset:
            base_images.append(img)
            self.labels.append(label)
            classes.append(label_to_class[label])

        aug_managers = []
        if self.args.use_canny:
            aug_managers.append(CannyAugmentationManager)
        if self.args.use_depth:
            aug_managers.append(DepthAugmentationManager)
        if self.args.use_seg:
            aug_managers.append(SegmentAugmentationManager)
        if self.args.use_color:
            aug_managers.append(ColorControlNetAugmentationManager)
        if self.args.use_nerf:
            aug_managers.append(NerfAugmentationManager)

        augmented_images = []
        augmented_labels = []
        for manager_class in aug_managers:
            manager = manager_class()
            augmented_images.extend(manager.generate_augmentations(base_images, classes))
            augmented_labels.extend(self.labels)

        self.images = base_images + augmented_images
        self.labels = self.labels + augmented_labels

        # here, we shuffle images and labels together
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class TreeAugmentedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.transform = transform
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label, class_name = self.dataset[index]
        return self.transform(img), label, class_name 

def create_datasets(args):
    root = './torch'

    # TODO make num_ways and num_shots come from args as well
    dataset_name = args.dataset
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
        data_dir = os.path.join(root, 'flowers102')
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

    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug_dataset = AugmentedDataset(base_dataset, new_label_to_class, transform=transform, args=args)
    classical_dataset = ClassicalDataset(base_dataset, basic_transform, duplicate_factor=len(aug_dataset)//len(base_dataset))
    test_dataset = ClassicalDataset(test_dataset, transform, duplicate_factor=1)

    print('Dataset sizes:')
    print(f'Classical dataset: {len(classical_dataset)}')
    print(f'Augmented dataset: {len(aug_dataset)}')
    print(f'Test dataset: {len(test_dataset)}')

    return aug_dataset, classical_dataset, test_dataset

if __name__ == '__main__':
    args = {
        'use_canny': True,
        'use_depth': True,
        'use_seg': True,
        'use_color': True,
        'use_nerf': True
    }
    # here we make args an object
    import argparse
    args = argparse.Namespace(**args)
    aug_dataset, base_dataset, test_dataset = create_datasets(args)
    # example_aug_dataset = CustomDataset(base_dataset, label_to_class, None, args=args)

    def save_random_samples(dataset, folder_name="custom_dataset_examples", num_samples=20):
        # Ensure the target folder exists
        os.makedirs(folder_name, exist_ok=True)
    
        # Randomly select samples
        indices = random.sample(range(len(dataset)), num_samples)
    
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            
            file_name = f"{folder_name}/sample_{i}_label_{label}.png"
            image.save(file_name)
            
            print(f"Saved: {file_name}")

    # print(len(example_aug_dataset))

    # save_random_samples(example_aug_dataset)