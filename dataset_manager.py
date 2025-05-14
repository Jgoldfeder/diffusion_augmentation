import os
import json
import random
import logging
from collections import defaultdict

import argparse
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import Caltech256, ImageFolder, StanfordCars, OxfordIIITPet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# we need to get the dataset, as well as the class name which corresponds to ea. label
def get_dataset_from_torch(dataset_name: str, root='./torch') -> tuple[Dataset, dict[int, str]]:
	if dataset_name == 'caltech256':
		try:
			dataset = Caltech256(root=root, download=False)
		except FileNotFoundError:
			raise Exception('Download dataset using commands in init_cloud_machine.sh in scripts folder.')
		label_to_class = dict()
		for category in dataset.categories:
			parts = category.split('.')
			label = int(parts[0]) - 1
			class_name = parts[1]
			label_to_class[label] = class_name
	elif dataset_name == 'flowers102':
		data_dir = os.path.join(root, 'flowers102')
		try:
			train_dataset = ImageFolder(os.path.join(data_dir, 'train'))
			validation_dataset = ImageFolder(os.path.join(data_dir, 'valid'))
		except FileNotFoundError:
			raise Exception('Download dataset from kaggle: https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset')
		dataset = ConcatDataset([train_dataset, validation_dataset])
		with open(os.path.join(data_dir, 'cat_to_name.json'), 'r') as f:
			label_str_to_class: dict = json.load(f)
		label_to_class = dict()
		for label_str, class_name in label_str_to_class.items():
			label_to_class[train_dataset.class_to_idx[label_str]] = class_name
	elif dataset_name == 'stanford_dogs':
		try:
			data_dir = os.path.join(root, 'stanford_dogs', 'Images')
			label_to_class = dict()
			label = 0
			dataset = ImageFolder(data_dir)
			for class_name, label in dataset.class_to_idx.items():
				label_to_class[label] = class_name.split('-')[1].lower()
		except FileNotFoundError:
			raise Exception('Download dataset by running wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
	elif dataset_name == 'stanford_cars':
		try:
			train_dataset = StanfordCars(root=root, download=False)
			test_dataset = StanfordCars(root=root, download=False, split="test")
		except FileNotFoundError:
			raise Exception('Download dataset from kaggle: https://www.kaggle.com/api/v1/datasets/download/rickyyyyyyy/torchvision-stanford-cars')
		dataset = ConcatDataset([train_dataset, test_dataset])
		label_to_class = dict()
		for i in range(len(train_dataset.classes)):
			label_to_class[i] = train_dataset.classes[i]
	elif dataset_name == 'food101':
		try:
			data_dir = os.path.join(root, 'food101', 'images')
			dataset = ImageFolder(data_dir)
			label_to_class = dict()
			for class_name, label in dataset.class_to_idx.items():
				label_to_class[label] = class_name
			print(label_to_class)
		except FileNotFoundError:
			raise Exception('Download dataset from kaggle: https://www.kaggle.com/datasets/dansbecker/food-101')
	elif dataset_name == 'oxford-iiit-pet':
		dataset = OxfordIIITPet(root=root, download=True)
		label_to_class = {label: class_name for class_name, label in dataset.class_to_idx.items()}
	else:
		raise Exception(f"Dataset {dataset_name} not supported")
	return dataset, label_to_class

def pick_random_labels(label_to_class: dict[int, str], num_ways: int):
	chosen_labels = sorted(random.sample(list(label_to_class.keys()), num_ways))
	chosen_classes = [label_to_class[label] for label in chosen_labels]
	logging.info(f"selected classes: {chosen_classes}")
	return chosen_labels

def partition_train_test(dataset: Dataset, chosen_labels: list[int], num_shots: int):
	labels_indexes = defaultdict(list)
	for i, (_, label) in enumerate(dataset):
		if label in chosen_labels:
			labels_indexes[label].append(i)

	train_indexes = []
	test_indexes = []
	for label, indexes in labels_indexes.items():
		random.shuffle(indexes)
		train_indexes.extend(indexes[:num_shots])
		test_indexes.extend(indexes[num_shots:])
	logging.info(f'num train: {len(train_indexes)}, num test: {len(test_indexes)}')

	train_dataset = Subset(dataset, train_indexes)
	test_dataset = Subset(dataset, test_indexes)

	return train_dataset, test_dataset

def get_dataset_path(dataset_name: str, num_ways: int, num_shots: int, subset: int, train: bool):
	file_path = os.path.join(
		'few_shot_datasets', 
 		dataset_name, 
		f"{num_ways}_ways",
  		f"{num_shots}_shot", 
   		f"subset_{subset}", 
    		'train' if train else 'test'
	)
	return file_path

def save_to_dir(dataset: Dataset, chosen_labels: list[int], label_to_class: dict[int, str], dataset_name: str, num_ways: int, num_shots: int, subset: int, train: bool):
	dataset_path = get_dataset_path(dataset_name, num_ways, num_shots, subset, train)
	os.makedirs(dataset_path, exist_ok=True)

	num_saved_by_label = defaultdict(int)
	for img, label in dataset:
		class_path = os.path.join(dataset_path, f'{chosen_labels.index(label)}_{label_to_class[label]}')
		os.makedirs(class_path, exist_ok=True)
		file_path = os.path.join(class_path, f"{num_saved_by_label[label]}.png")
		img.save(file_path)
		num_saved_by_label[label] += 1

def create_fewshot_dataset(dataset_name: str, num_ways: int, num_shots: int, subset: int):
	original_dataset, label_to_class = get_dataset_from_torch(dataset_name)
	chosen_labels = pick_random_labels(label_to_class, num_ways)
	train_dataset, test_dataset = partition_train_test(original_dataset, chosen_labels, num_shots)
	save_to_dir(train_dataset, chosen_labels, label_to_class, dataset_name, num_ways, num_shots, subset, train=True)
	save_to_dir(test_dataset, chosen_labels, label_to_class, dataset_name, num_ways, num_shots, subset, train=False)

# NOTE this transform is specific for resnet, might want to use another for a diff model
def get_base_transform():
	return Compose([
		Resize(size=(224, 224)),
		ToTensor(),
		Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	])

# TODO try this before returning images from the FolderDataset
def get_classical_transform():
	return Compose([
		Resize(size=(256, 256)),
		RandomCrop(size=(224, 224)),
		ColorJitter(
			brightness=0.4,
			contrast=0.4,
			saturation=0.4,
			hue=0.2 
		),
		RandomHorizontalFlip(p=0.5),
		RandomVerticalFlip(p=0.5), 
		RandomRotation(degrees=10)
	])

class FolderDataset(Dataset):
	def __init__(self, dataset_path):
		self.images: list[Image.Image] = []
		self.labels: list[int] = []
		self.labels_to_class: dict[int, str] = dict()
		self.base_transform = get_base_transform()
		self.dataset_path = dataset_path

		for local_class_path in os.listdir(dataset_path):
			class_parts = local_class_path.split('_')
			class_label = int(class_parts[0])
			class_name = '_'.join(class_parts[1:])
			self.labels_to_class[class_label] = class_name

			class_path = os.path.join(dataset_path, local_class_path)

			for img in os.listdir(class_path):
				if img.endswith('.png'):
					img = Image.open(os.path.join(class_path, img)).convert('RGB').resize((224, 224))
					self.images.append(img)
					self.labels.append(class_label)

	def get_class_for_label(self, label):
		return self.labels_to_class[label]

	def get_class_for_index(self, index):
		return self.get_class_for_label(self.labels[index])

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.base_transform(self.images[index]), self.labels[index]

class SplitDataset(FolderDataset):
	def __init__(self, dataset_path, indexes_to_keep):
		super().__init__(dataset_path)
		self.images = [self.images[i] for i in indexes_to_keep]
		self.labels = [self.labels[i] for i in indexes_to_keep]

class ClassicalDataset(FolderDataset):
	def __init__(self, dataset_path, duplicate_factor=1):
		super().__init__(dataset_path)
		self.duplicate_factor = duplicate_factor

	def __len__(self):
		return len(self.images) * (self.duplicate_factor)

	def __getitem__(self, index):
		true_index = index % len(self.images)
		img = self.images[true_index]
		label = self.labels[true_index]
		transformed_img = self.base_transform(get_classical_transform()(img))
		return transformed_img, label


def split_train_val(dataset: FolderDataset):
	splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5)
	train_indices, val_indices = next(splitter.split(dataset, dataset.labels))
	return SplitDataset(dataset.dataset_path, train_indices), SplitDataset(dataset.dataset_path, val_indices)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--num_ways', type=int, required=True)
	parser.add_argument('--num_shots', type=int, required=True)
	parser.add_argument('--subset', type=int, required=True)
	return parser.parse_args()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	args = parse_args()
	dataset_name = args.dataset
	num_ways = args.num_ways
	num_shots = args.num_shots
	subset = args.subset

	random.seed(args.subset)

	create_fewshot_dataset(dataset_name, num_ways, num_shots, subset)
