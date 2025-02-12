import os
import json
import random
import logging
from collections import defaultdict

import argparse
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import Caltech256, ImageFolder

# we need to get the dataset, as well as the class name which corresponds to ea. label
def get_dataset_from_torch(dataset_name: str, root='./torch') -> tuple[Dataset, dict[int, str]]:
	if dataset_name == 'caltech256':
		dataset = Caltech256(root=root, download=True)
		label_to_class = dict()
		for category in dataset.categories:
			parts = category.split('.')
			label = int(parts[0]) - 1
			class_name = parts[1]
			label_to_class[label] = class_name
	elif dataset_name == 'flowers102':
		data_dir = os.path.join(root, 'flower_data')
		try:
			train_dataset = ImageFolder(os.path.join(data_dir, 'train'))
			validation_dataset = ImageFolder(os.path.join(data_dir, 'valid'))
		except FileNotFoundError:
			raise Exception('Download the dataset from kaggle from the following page using curl: https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset')
		dataset = ConcatDataset([train_dataset, validation_dataset])
		with open(os.path.join(data_dir, 'cat_to_name.json'), 'r') as f:
			label_str_to_class: dict = json.load(f)
		label_to_class = dict()
		for label_str, class_name in label_str_to_class.items():
			label_to_class[int(label_str)] = class_name
	else:
		raise Exception(f"Dataset {dataset_name} not supported")
	return dataset, label_to_class

def pick_random_labels(label_to_class: dict[int, str], num_ways: int):
	chosen_labels = sorted(random.sample(list(label_to_class.keys()), num_ways))
	chosen_classes = [label_to_class[label] for label in chosen_labels]
	logging.info(f"selected classes: {chosen_classes}")
	return chosen_labels

def split_dataset(dataset: Dataset, chosen_labels: list[int], num_shots: int):
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
	logging.debug(f'num train: {len(train_indexes)}, num test: {len(test_indexes)}')

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
	train_dataset, test_dataset = split_dataset(original_dataset, chosen_labels, num_shots)
	save_to_dir(train_dataset, chosen_labels, label_to_class, dataset_name, num_ways, num_shots, subset, train=True)
	save_to_dir(test_dataset, chosen_labels, label_to_class, dataset_name, num_ways, num_shots, subset, train=False)

def create_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--num_ways', type=int, required=True)
	parser.add_argument('--num_shots', type=int, required=True)
	parser.add_argument('--subset', type=int, required=True)
	return parser

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	parser = create_parser()
	args = parser.parse_args()

	dataset_name = args.dataset
	num_ways = args.num_ways
	num_shots = args.num_shots
	subset = args.subset

	random.seed(args.subset)

	create_fewshot_dataset(dataset_name, num_ways, num_shots, subset)