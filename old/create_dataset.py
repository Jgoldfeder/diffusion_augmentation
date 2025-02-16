import random
import os
from collections import defaultdict
import json
import torch
import argparse

from torchvision.datasets import Caltech256, ImageFolder
from torch.utils.data import Dataset, Subset


class RemappedDataset(Dataset):
    def __init__(self, dataset, old_to_new_labels):
        self.dataset = dataset
        self.old_to_new_labels = old_to_new_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.convert('RGB'), self.old_to_new_labels[label]

def get_label_remapping(old_labels_set):
    old_to_new_labels = dict()
    old_labels_sorted = sorted(list(set(old_labels_set)))
    for i in range(len(old_labels_sorted)):
        old_to_new_labels[old_labels_sorted[i]] = i
    return old_to_new_labels

def split_train_test(dataset, class_to_label, labels, num_ways, num_shots, subset):
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

    # TODO: fix this to remove the RemappedDataset class

    old_to_new_labels = get_label_remapping(set(target_labels))
    train_dataset = RemappedDataset(Subset(dataset, train_indexes), old_to_new_labels)
    test_dataset = RemappedDataset(Subset(dataset, test_indexes), old_to_new_labels)

    print(len(train_dataset), len(test_dataset))

    return train_dataset, test_dataset, old_to_new_labels

def create_list_from_dataset(dataset, new_to_old_labels, label_to_class):
     dataset_list = []
     for i in range(len(dataset)):
          img, label = dataset[i]
          class_name = label_to_class[new_to_old_labels[label]]
          dataset_list.append((img, label, class_name))
     return dataset_list

def save_to_dir(dataset, dataset_name, num_shots, subset, train=True):
    file_path = os.path.join('few_shot_datasets', 
                            dataset_name, 
                            f"{num_shots}_shot", 
                            f"subset_{subset}", 
                            'train' if train else 'test')
    
    os.makedirs(file_path, exist_ok=True)

    #dataset is a list of tuples (image, label, class_name)
    for img, label, class_name in dataset:
        folder_name = f"{label}_{class_name}" #folder name is {label}_{class_name}
        os.makedirs(os.path.join(file_path, folder_name), exist_ok=True)

        #use number of images in folder to name the image
        img_count = len(os.listdir(os.path.join(file_path, folder_name)))
        img.save(os.path.join(file_path, folder_name, f"{img_count}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create few-shot dataset splits')
    parser.add_argument('--dataset', type=str, default='caltech256', choices=['caltech256', 'flowers102'],
                        help='Dataset to use (default: caltech256)')
    parser.add_argument('--subset', type=int, default=41,
                        help='which subset to create (default: 41)')
    parser.add_argument('--num_ways', type=int, default=5,
                        help='Number of ways/classes (default: 5)')
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of shots/examples per class (default: 5)')
    
    args = parser.parse_args()
    
    print('main func called')
    
    dataset_name = args.dataset
    subset = args.subset
    num_ways = args.num_ways
    num_shots = args.num_shots

    random.seed(subset)
    
    if dataset_name == 'caltech256':
        dataset = Caltech256(root='./torch', download=True)
        class_to_label = dict()
        label_to_class = dict()
        for category in dataset.categories:
            parts = category.split('.')
            label = int(parts[0]) - 1
            class_name = parts[1]
            class_to_label[class_name] = label
            label_to_class[label] = class_name
        labels = [label for _, label in dataset]
    elif dataset_name == 'flowers102':
        root = './torch'
        data_dir = os.path.join(root, 'flowers102')
        #try:
        train_dataset = ImageFolder(os.path.join(data_dir, 'train'))
        validation_dataset = ImageFolder(os.path.join(data_dir, 'valid'))
        dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])
        with open(os.path.join(data_dir, 'cat_to_name.json'), 'r') as f:
            label_to_class_as_str = json.load(f)
        label_to_class = {int(label_str): class_name for label_str, class_name in label_to_class_as_str.items()}
        class_to_label = {class_name: label for label, class_name in label_to_class.items()}
        labels = [int(target) for target in train_dataset.targets]
        # except FileNotFoundError:
        #     print('Download the dataset from kaggle from the following page using curl:')
        #     print('https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset')
        #     exit(1)

    train_dataset, test_dataset, old_to_new_labels = split_train_test(dataset, class_to_label, labels, num_ways, num_shots, subset=subset)
    new_to_old_labels = {v: k for k, v in old_to_new_labels.items()}

    train_dataset_list = create_list_from_dataset(train_dataset, new_to_old_labels, label_to_class)
    test_dataset_list = create_list_from_dataset(test_dataset, new_to_old_labels, label_to_class)

    save_to_dir(train_dataset_list, dataset_name, num_shots, subset, train=True)
    save_to_dir(test_dataset_list, dataset_name, num_shots, subset, train=False)

