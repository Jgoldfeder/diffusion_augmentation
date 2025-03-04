import wandb

from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode, AugmentationType, ProbabilityLimits

class TreeAugmentedDatasetWithClassical(TreeAugmentedDataset):
	def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
		super().__init__(dataset_path, augmentation_tree, num_augmentations_per_image)

	def __getitem__(self, index):
		img = self.images[index]
		label = self.labels[index]
		return dataset_manager.get_base_transform()(dataset_manager.get_classical_transform()(img)), label

if __name__ == '__main__':
	import logging
	import dataset_manager
	import network_model
	from genetic_algorithm import genome_to_tree
	from network_model import ModelType
	import torch
	import random
	import time

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	dataset = 'oxford-iiit-pet'
	num_ways = 5
	num_shots = 2
	subset = 44

	num_augmentations_per_image = 5

	# tree_genome = [AugmentationType.get_random_augmentation().value, ProbabilityLimits.get_random_probability(), AugmentationType.get_random_augmentation().value, .5, AugmentationType.get_random_augmentation().value, .5]
	tree_genome = [3, 0.4520504900156664, 3, .5, 1, .5]
	node = genome_to_tree(tree_genome)
	print(str(node))

	train_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=True)
	test_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=False)

	for i in range(20):
		seed = 41 + i
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

		# node = BinaryAugmentationNode()
		# node.make_random_tree(2)
		# print(str(node))

		if i % 2:
			train_dataset = TreeAugmentedDataset(train_path, node, num_augmentations_per_image)
		else:
			train_dataset = TreeAugmentedDatasetWithClassical(train_path, node, num_augmentations_per_image)
		test_dataset = FolderDataset(test_path)

		model = network_model.get_model_for_finetune(ModelType.RESNET50, num_ways)
		model_results = network_model.train_and_test(model, train_dataset, test_dataset, 200, 'cuda')
		print(model_results)

		with open('temp.txt', 'a') as f:
			if i % 2:
				f.write('without classical\n')
			else:
				f.write('with classical\n')
			f.write(str(model_results.accs[-2:]) + '\n')

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


