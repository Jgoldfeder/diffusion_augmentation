import wandb
import argparse
from dataset_manager import FolderDataset
from augmentation_tree import TreeAugmentedDataset, BinaryAugmentationNode, AugmentationType, ProbabilityLimits
from network_model import ModelType
import os
import logging
import dataset_manager
import network_model
from genetic_algorithm import genome_to_tree
import torch
import random
import time
from transformers import AutoImageProcessor
from torchvision import transforms

class TreeAugmentedDatasetWithClassical(TreeAugmentedDataset):
	def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int, image_processor=None):
		super().__init__(dataset_path, augmentation_tree, num_augmentations_per_image, image_processor)

	def __getitem__(self, index):
		img = self.images[index]
		label = self.labels[index]
		return dataset_manager.get_base_transform()(dataset_manager.get_classical_transform()(img)), label

def parse_args():
	parser = argparse.ArgumentParser(description='Test tree accuracy with different models and datasets')
	parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., oxford-iiit-pet, caltech256)')
	parser.add_argument('--num_ways', type=int, required=True, help='Number of ways (classes)')
	parser.add_argument('--num_shots', type=int, required=True, help='Number of shots (examples per class)')
	parser.add_argument('--model_type', type=str, required=True, 
					  choices=['resnet50', 'vit224', 'mobilenetv2', 'vits'], 
					  help='Model type to use (resnet50: standard CNN, vit224: Vision Transformer, mobilenetv2: lightweight CNN, vits: small Vision Transformer)')
	parser.add_argument('--subset', type=int, default=44, help='Subset of classes to use')
	parser.add_argument('--num_augmentations', type=int, default=2, help='Number of augmentations per image')
	parser.add_argument('--num_runs', type=int, default=6, help='Number of runs to perform')
	parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
	parser.add_argument('--seed_start', type=int, default=41, help='Starting seed for random number generation')
	parser.add_argument('--genome', type=str, required=True, 
					  help='Augmentation tree genome as 6 comma-separated numbers (e.g., "3,0.3,6,0.4,6,0.6")')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	# Parse genome from command line
	try:
		tree_genome = [float(x) for x in args.genome.split(',')]
		if len(tree_genome) != 6:
			raise ValueError("Genome must contain exactly 6 numbers")
	except ValueError as e:
		logging.error(f"Invalid genome format: {e}")
		logging.error("Genome must be 6 comma-separated numbers (e.g., '3,0.3,6,0.4,6,0.6')")
		exit(1)

	node = genome_to_tree(tree_genome)
	print(f"Using augmentation tree: {str(node)}")

	train_path = dataset_manager.get_dataset_path(args.dataset, args.num_ways, args.num_shots, args.subset, train=True)
	test_path = dataset_manager.get_dataset_path(args.dataset, args.num_ways, args.num_shots, args.subset, train=False)

	# Initialize image processor for ViT-Small if needed
	image_processor = None
	if args.model_type == 'vits':
		image_processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")

	for i in range(args.num_runs):
		seed = args.seed_start + i
		wandb.init(
			project="learned-tree-tests",
			config={
				"subset": args.subset,
				"num_shots": args.num_shots,
				"dataset": args.dataset,
				"num_ways": args.num_ways,
				"model_type": args.model_type,
				"seed": seed,
				"without_classical": i % 2
			}
		)
		random.seed(seed)

		if i % 2:
			train_dataset = TreeAugmentedDataset(train_path, node, args.num_augmentations, image_processor=image_processor)
		else:
			train_dataset = TreeAugmentedDatasetWithClassical(train_path, node, args.num_augmentations, image_processor=image_processor)
		test_dataset = FolderDataset(test_path, image_processor=image_processor)

		model = network_model.get_model_for_finetune(ModelType(args.model_type), args.num_ways)
		model_results = network_model.train_and_test(model, train_dataset, test_dataset, args.num_epochs, 'cuda')
		print(f"Run {i+1}/{args.num_runs} Results:")
		print(model_results)

		with open('temp.txt', 'a') as f:
			f.write(f"Run {i+1} - {'without' if i % 2 else 'with'} classical augmentation\n")
			f.write(f"Model: {args.model_type}, Dataset: {args.dataset}, Ways: {args.num_ways}, Shots: {args.num_shots}\n")
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
			'tree': str(node),
			'model_type': args.model_type,
			'best_test_accuracy': max(results.accs)
		})
		wandb.finish()


