import random
import os
from collections import defaultdict

from torchvision.datasets import Caltech256
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

def split_train_test(dataset, class_to_label, labels, num_ways, num_shots, seed):
    random.seed(seed)
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

def create_list_from_dataset(dataset, new_to_old_labels, label_to_class):
     dataset_list = []
     for i in range(len(dataset)):
          img, label = dataset[i]
          class_name = label_to_class[new_to_old_labels[label]]
          dataset_list.append((img, label, class_name))
     return dataset_list

if __name__ == '__main__':
	print('main func called')
	# read in caltech 256 dataset

	dataset_name = 'caltech256'
	seed = 42
	num_ways = 5 # will always be 5
	num_shots = 2

	dataset = Caltech256(root='./torch')

	class_to_label = dict()
	label_to_class = dict()
	for category in dataset.categories:
		parts = category.split('.')
		label = int(parts[0]) - 1
		class_name = parts[1]
		class_to_label[class_name] = label
		label_to_class[label] = class_name
	labels = [label for _, label in dataset]

	train_dataset, test_dataset, old_to_new_labels = split_train_test(dataset, class_to_label, labels, num_ways, num_shots, seed=seed)
	new_to_old_labels = {v: k for k, v in old_to_new_labels.items()}
	dataset_list = create_list_from_dataset(train_dataset, new_to_old_labels, label_to_class)

	breakpoint()
     
def save_to_dir(dataset, dataset_name, num_shots, seed, train=True):
    file_path = os.path.join('few_shot_datasets', 
                            dataset_name, 
                            num_shots, 
                            seed, 
                            'train' if train else 'test')
    
    os.makedirs(file_path, exist_ok=True)

    #dataset is a list of tuples (image, label, class_name)
    for img, label, class_name in dataset:
        folder_name = f"{label}_{class_name}" #folder name is {label}_{class_name}
        os.makedirs(os.path.join(file_path, folder_name), exist_ok=True)

        #use number of images in folder to name the image
        img_count = len(os.listdir(os.path.join(file_path, folder_name)))
        img.save(os.path.join(file_path, folder_name, f"{img_count}.png"))