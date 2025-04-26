import wandb
import argparse

from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode, AugmentationType, ProbabilityLimits

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class TreeAugmentedDatasetWithClassical(TreeAugmentedDataset):
	def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
		super().__init__(dataset_path, augmentation_tree, num_augmentations_per_image)

	def __getitem__(self, index):
		img = self.images[index]
		label = self.labels[index]
		return dataset_manager.get_base_transform()(dataset_manager.get_classical_transform()(img)), label

def main():
	parser = argparse.ArgumentParser(description='Train model from scratch on a dataset')
	parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset to use (default: caltech256)')
	parser.add_argument('--num_ways', type=int, default=10, help='Number of classes (default: 10)')
	parser.add_argument('--num_shots', type=int, default=2, help='Number of shots per class (default: 2)')
	parser.add_argument('--subset', type=int, default=42, help='Subset number (default: 42)')
	args = parser.parse_args()

	dataset = args.dataset
	num_ways = args.num_ways
	num_shots = args.num_shots
	subset = args.subset

	import logging
	import dataset_manager
	import network_model
	from genetic_algorithm import genome_to_tree
	from network_model import ModelType
	import torch
	import random
	import time

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	num_augmentations_per_image = 2

	tree_genome = [6,0.3,4,0.6319112651020988,3,0.5918927146775271]
	node = genome_to_tree(tree_genome)
	print(str(node))

	train_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=True)
	test_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=False)

	for i in range(6):
		seed = 42 + i
		wandb.init(
			project="learned-tree-tests",
			config={
				"subset": subset,
				"num_shots": num_shots,
				"dataset": dataset,
				"num_ways": num_ways,
				"seed": seed,
				"without_classical": i % 2
			}
		)
		random.seed(seed)
        
		if i % 2:
			train_dataset = TreeAugmentedDatasetWithClassical(train_path, node, num_augmentations_per_image)
		else:
			train_dataset = TreeAugmentedDataset(train_path, node, num_augmentations_per_image)
		test_dataset = FolderDataset(test_path)

		model = network_model.get_model_for_scratch(ModelType.RESNET50, num_ways)
		model_results = network_model.train_and_test(model, train_dataset, test_dataset, 200, 'cuda:2', finetune=False)
		print(model_results)

		# with open('temp.txt', 'a') as f:
		# 	if i % 2:
		# 		f.write('without classical\n')
		# 	else:
		# 		f.write('with classical\n')
		# 	f.write(str(model_results.accs[-6:]) + '\n')

		results = model_results
		for (train_loss, train_acc, test_loss, test_acc) in zip(results.train_losses, results.train_accs, results.losses, results.accs):
			wandb.log({
				"train_loss": train_loss,
				"train_accuracy": train_acc,
				"test_loss": test_loss,
				"test_accuracy": test_acc
			})
		wandb.log({
			'tree': str(node)
		})
		wandb.finish()

if __name__ == '__main__':
	main()