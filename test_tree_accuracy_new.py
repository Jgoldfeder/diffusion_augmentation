from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode

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

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	dataset = 'caltech256'
	num_ways = 5
	num_shots = 2
	subset = 42

	num_augmentations_per_image = 1

	tree_genome = [1, .5]
	node = genome_to_tree(tree_genome)
	print(str(node))

	train_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=True)
	test_path = dataset_manager.get_dataset_path(dataset, num_ways, num_shots, subset, train=False)

	train_dataset = TreeAugmentedDataset(train_path, node, num_augmentations_per_image)
	test_dataset = FolderDataset(test_path)

	model = network_model.get_model_for_finetune(ModelType.RESNET50, num_ways)
	model_results = network_model.train_and_test(model, train_dataset, test_dataset, 20, 'cuda')
	print(model_results)


