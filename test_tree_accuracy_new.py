import wandb
import logging
import dataset_manager
import network_model
from genetic_algorithm import genome_to_tree
from network_model import ModelType
import torch
import random
import time


from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode, AugmentationType, ProbabilityLimits

class TreeAugmentedDatasetWithClassical(TreeAugmentedDataset):
	def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
		super().__init__(dataset_path, augmentation_tree, num_augmentations_per_image)

	def __getitem__(self, index):
		img = self.images[index]
		label = self.labels[index]
		return dataset_manager.get_base_transform()(dataset_manager.get_classical_transform()(img)), label

def run_test(dataset, num_ways, num_shots, subset, tree_genome, seed, without_classical, model_type: ModelType):
	train_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=True)
	test_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=False)

	wandb.init(
		project="iclr_best_genomes",
		config={
			"subset": subset,
			"num_shots": num_shots,
			"dataset": dataset,
			"num_ways": num_ways,
			"seed": seed,
			"without_classical": without_classical,
			"model_type": model_type.value
		}
	)
	random.seed(seed)

	node = genome_to_tree(tree_genome)
	print(str(node))

	if without_classical:
		train_dataset = TreeAugmentedDataset(train_path, node, num_augmentations_per_image)
	else:
		train_dataset = TreeAugmentedDatasetWithClassical(train_path, node, num_augmentations_per_image)
	test_dataset = FolderDataset(test_path)

	model = network_model.get_model_for_finetune(model_type, num_ways)
	model_results = network_model.train_and_test(model, train_dataset, test_dataset, 600, 'cuda')
	print(model_results)

	with open('temp.txt', 'a') as f:
		if without_classical:
			f.write('without classical\n')
		else:
			f.write('with classical\n')
		f.write(str(model_results.accs[-6:]) + '\n')

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
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	model_type = ModelType.VITS
	num_ways = 5
	num_shots = 1
	num_augmentations_per_image = 2
	dsets = [
		# {
		# 	'dataset': 'caltech256',
		# 	'subset': 43,
		# },
		# {
		# 	'dataset': 'flowers102',
		# 	'subset': 50,
		# },
		# {
		# 	'dataset': 'stanford_dogs',
		# 	'subset': 46,
		# },
		# {
		# 	'dataset': 'stanford_cars',
		# 	'subset': 44,
		# },
		# {
		# 	'dataset': 'oxford-iiit-pet',
		# 	'subset': 44,
		# },
		# {
		# 	'dataset': 'food101',
		# 	'subset': 43,
		# }
	]

	for dset in dsets:
		dataset = dset['dataset']
		subset = dset['subset']
		for i in range(0, 20, 1):
			seed = 41 + i
			# random_prob = ProbabilityLimits.get_random_probability()
			# random_nodes = [random.choice([5, 6]) for _ in range(3)]
			# tree_genome = [random_nodes[0], random_prob, random_nodes[1], .5, random_nodes[2], .5]
			tree_genome = [6, 0.3861255, 6, 0.43463782, 6, 0.64035153]
			run_test(dataset, num_ways, num_shots, subset, tree_genome, seed, 0, model_type)
			run_test(dataset, num_ways, num_shots, subset, tree_genome, seed, 1, model_type)
		


