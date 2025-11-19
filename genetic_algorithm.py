import random
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import wandb
import pygad
import logging
import argparse
import torch
import numpy as np

import network_model
import dataset_manager
from network_model import ModelResults, ModelType
from dataset_manager import FolderDataset
from augmentation_tree import BinaryAugmentationNode, AugmentationType, TreeAugmentedDataset, TreeAugmentedDatasetFromDataset, ProbabilityLimits

from torch.utils.data import DataLoader
from torch import nn
import torchvision
from sklearn.metrics import silhouette_score

import timm

def genome_to_tree(genome, curr_index=0) -> BinaryAugmentationNode:
	if curr_index >= len(genome):
		return None
	node = BinaryAugmentationNode()
	node.augmentation_type = AugmentationType(int(genome[curr_index]))
	node.left_probability = genome[curr_index + 1]
	children_start = (curr_index + 1) * 2 
	node.left = genome_to_tree(genome, children_start)
	node.right = genome_to_tree(genome, children_start + 2)
	return node

def genome_to_number(genome):
    number = 0
    mult = 1
    for i in range(0, len(genome), 2):
        number += (genome[i] + genome[i+1]) * mult
        mult *= 10
    return number

def gene_space(tree_depth):
    gene_space = []
    for i in range(2 ** tree_depth - 1):
        gene_space.extend([[i for i in range(len(AugmentationType))], {"low": ProbabilityLimits.LOW.value, "high": ProbabilityLimits.HIGH.value}])
    return gene_space

def initial_population(sol_per_pop, tree_depth):
	population = []
	augmentation_cycler = 0
	num_genes = 2 * (2 ** tree_depth - 1)
	added_baseline = False

	for i in range(sol_per_pop):
		genome = []
		for j in range(0, num_genes, 2):
			aug_type = AugmentationType.get_random_augmentation().value
			left_prob = ProbabilityLimits.get_random_probability()
			genome.extend([aug_type, left_prob])
		genome[0] = augmentation_cycler

		# have one tree that is our baseline
		if not added_baseline and AugmentationType.CLASSICAL.value == augmentation_cycler:
			added_baseline = True
			for j in range(2, num_genes, 2):
				genome[j] = AugmentationType.NONE.value

		augmentation_cycler += 1
		augmentation_cycler %= len(AugmentationType)
		population.append(genome)
	return population

class GAHelper:
	def __init__(
		self, dataset_name: str, num_ways: int, num_shots: int, subset: int,
		model_type: ModelType, tree_depth: int, num_augmentations_per_image: int,
		num_iterations_for_val: int, num_iterations_for_test: int, clustering_alpha: float,
		device
	):
		self.dataset_name = dataset_name
		self.num_ways = num_ways
		self.num_shots = num_shots
		self.subset = subset
		self.model_type = model_type
		self.tree_depth = tree_depth
		self.num_augmentations_per_image = num_augmentations_per_image
		self.num_iterations_for_val = num_iterations_for_val
		self.num_iterations_for_test = num_iterations_for_test
		self.clustering_alpha = clustering_alpha
		self.device = device

		self.train_path = dataset_manager.get_dataset_path(self.dataset_name, self.num_ways, self.num_shots, self.subset, train=True)
		self.test_path = dataset_manager.get_dataset_path(self.dataset_name, self.num_ways, self.num_shots, self.subset, train=False)

		self.tree_evals_per_generation = [0]
		self.fitness_cache: dict[float, float] = dict()

	def fitness_func(self, genome, num_shots):
		genome_number = genome_to_number(genome)
		if genome_number in self.fitness_cache:
			logging.info('fitness function call using cached fitness')
			return self.fitness_cache[genome_number]
		self.tree_evals_per_generation[-1] += 1

		node = genome_to_tree(genome)
		dataset = FolderDataset(self.train_path)
		train_dataset, val_dataset = dataset_manager.split_train_val(dataset)

		#fold 1
		tree_augmented_train_dataset = TreeAugmentedDatasetFromDataset(train_dataset, node, self.num_augmentations_per_image)
		model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
		model_results: ModelResults = network_model.train_and_val(model, tree_augmented_train_dataset, val_dataset, self.num_iterations_for_val, self.device)
		loss_fold1 = model_results.losses[-1]

		#fold 2 -- swap train and val
		tree_augmented_val_dataset = TreeAugmentedDatasetFromDataset(val_dataset, node, self.num_augmentations_per_image)
		model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
		model_results: ModelResults = network_model.train_and_val(model, tree_augmented_val_dataset, train_dataset, self.num_iterations_for_val, self.device)
		loss_fold2 = model_results.losses[-1]

		fitness = -1 * (loss_fold1 + loss_fold2) / 2
		logging.info(f'Fitness Score for {str(genome)}: {fitness}')
		self.fitness_cache[genome_number] = fitness
		return fitness

	def fitness_func_one_shot_training_loss(self, genome, num_shots):
		genome_number = genome_to_number(genome)
		if genome_number in self.fitness_cache:
			logging.info('fitness function call using cached fitness')
			return self.fitness_cache[genome_number]
		self.tree_evals_per_generation[-1] += 1
		
		node = genome_to_tree(genome)
		dataset = FolderDataset(self.train_path)
		tree_augmented_train_dataset = TreeAugmentedDatasetFromDataset(dataset, node, self.num_augmentations_per_image)
		model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
		model_results: ModelResults = network_model.train(model, tree_augmented_train_dataset, self.num_iterations_for_val, self.device)
		loss = model_results.train_losses[-1]
		logging.info(f'Fitness Score for {str(genome)}: {loss}')
		self.fitness_cache[genome_number] = loss
		return loss

	def fitness_func_one_shot_clustering(self, genome, num_shots):
		genome_number = genome_to_number(genome)
		if genome_number in self.fitness_cache:
			logging.info('fitness function call using cached fitness')
			return self.fitness_cache[genome_number]
		self.tree_evals_per_generation[-1] += 1
		
		node = genome_to_tree(genome)
		dataset = FolderDataset(self.train_path)
		tree_augmented_train_dataset = TreeAugmentedDatasetFromDataset(dataset, node, self.num_augmentations_per_image)

		data_loader = DataLoader(tree_augmented_train_dataset, batch_size=256, shuffle=False, num_workers=2)

		# model = torchvision.models.resnet50(pretrained=True)
		# model.fc = nn.Identity()
		
		model = timm.create_model("vit_base_patch16_224", pretrained=True)
		if hasattr(model, "head"):
			model.head = nn.Identity()
		elif hasattr(model, "classifier"):
			model.classifier = nn.Identity()

		embeddings = []
		true_labels = []

		with torch.no_grad():
			for (images, labels) in data_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				features = model(images)
				embeddings.append(features.cpu().numpy())
				true_labels.append(labels.cpu().numpy())

		embeddings = np.concatenate(embeddings, axis=0)
		true_labels = np.concatenate(true_labels, axis=0)

		def compute_cluster_radii(embeddings, clusters):
			"""
			For each cluster, compute the average distance from the cluster centroid
			to its points as a measure of the cluster's "radius."
			"""
			unique_clusters = np.unique(clusters)
			cluster_radii = {}
			for cluster in unique_clusters:
				indices = np.where(clusters == cluster)[0]
				points = embeddings[indices]
				centroid = np.mean(points, axis=0)
				distances = np.linalg.norm(points - centroid, axis=1)
				avg_radius = np.mean(distances)
				cluster_radii[cluster] = avg_radius
			return cluster_radii

		clusters_true = true_labels
		sil_true = silhouette_score(embeddings, clusters_true)
		cluster_radii_true = compute_cluster_radii(embeddings, clusters_true)
		avg_radius_true = np.mean(list(cluster_radii_true.values()))
		alpha = self.clustering_alpha

		fitness_score = alpha * sil_true - (1 - alpha) * (1 / (avg_radius_true + 1)) + 2

		logging.info(f'Fitness Score for {str(genome)}: {fitness_score}')
		self.fitness_cache[genome_number] = fitness_score
		return fitness_score

	def on_generation(self, ga_instance):
		best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
		best_genome = best_solution[0]
		best_fitness = best_solution[1]
		best_tree = genome_to_tree(best_genome)

		num_generations_finished = len(self.tree_evals_per_generation)
		num_fitness_evals = self.tree_evals_per_generation[-1]

		self.tree_evals_per_generation.append(0)

		logging.info(f'Generation finished: {num_generations_finished}:')
		logging.info(f'Num fitness evals: {num_fitness_evals}')
		logging.info(f'Best tree genome: {best_genome}')
		logging.info(str(best_tree))

		wandb.log({
			"generation": num_generations_finished,
			"best_fitness": best_fitness,
			"population_fitness_mean": np.mean(ga_instance.last_generation_fitness),
			"population_fitness_std": np.std(ga_instance.last_generation_fitness),
			"best_tree_genome": best_genome
		})

	def on_stop(self, ga_instance):
		best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
		best_tree = genome_to_tree(best_solution[0])

		train_dataset = TreeAugmentedDataset(self.train_path, best_tree, self.num_augmentations_per_image)
		test_dataset = FolderDataset(self.test_path)

		for i in range(41, 43):
			random.seed(i)

			model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
			model_results: ModelResults = network_model.train_and_test(model, train_dataset, test_dataset, self.num_iterations_for_test, self.device)
			best_tree_accuracy = model_results.accs[-1]

			logging.info(f'Best tree accuracy: {best_tree_accuracy}')

			for j in range(self.num_iterations_for_test):
				wandb.log({
					f'train_loss': model_results.train_losses[j],
					f'train_acc': model_results.train_accs[j],
					f'test_loss': model_results.losses[j],
					f'test_acc': model_results.accs[j],
					'epoch': j
				})
			wandb.log({
				"confusion_matrix": wandb.plot.confusion_matrix(
					probs=None,
					y_true=model_results.labels,
					preds=model_results.preds,
					class_names=[test_dataset.labels_to_class[i] for i in range(self.num_ways)]
				)
			})
			wandb.log({
				"best_tree_accuracy": best_tree_accuracy
			})

def parse_args():
    parser = argparse.ArgumentParser(description='Run genetic algorithm for augmentation tree optimization')
    parser.add_argument('--num_generations', type=int, default=3, help='Number of generations')
    parser.add_argument('--sol_per_pop', type=int, default=10, help='Solutions per population')
    parser.add_argument('--num_parents_mating', type=int, default=4, help='Number of parents for mating')
    parser.add_argument('--keep_elitism', type=int, default=1, help='Number of elites to keep')
    parser.add_argument('--keep_parents', type=int, default=4, help='Number of parents to keep')
    parser.add_argument('--mutation_percent', type=int, default=10, help='Mutation percentage')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num_ways', type=int, required=True, help='Number of ways (classes)')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots (examples per class)')
    parser.add_argument('--subset', type=int, required=True, help='Which subset of classes to use')
    parser.add_argument('--model_type', type=str, default='resnet50', help='Which base model to use')
    parser.add_argument('--tree_depth', type=int, required=True, help='Depth of the augmentation tree')
    parser.add_argument('--num_augmentations_per_image', type=int, default=5, help='Number of augmentations per image to expand dataset by')
    parser.add_argument('--num_iterations_for_val', type=int, default=20, help='Number of iterations to train each tree before getting loss from val')
    parser.add_argument('--num_iterations_for_test', type=int, default=400, help='Number of iterations to train best tree before final testing')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--one_shot_training_loss', type=bool, default=False, help='Whether to use one shot training loss')
    parser.add_argument('--one_shot_clustering', type=bool, default=False, help='Whether to use one shot clustering')
    parser.add_argument('--clustering_alpha', type=float, default=0.5, help='Alpha parameter for clustering fitness')
    return parser.parse_args()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	args = parse_args()

	# might also need to seed numpy here too
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	ga_helper = GAHelper(
		dataset_name=args.dataset,
		num_ways=args.num_ways,
		num_shots=args.num_shots,
		subset=args.subset,
		model_type=ModelType(args.model_type),
		tree_depth=args.tree_depth,
		num_augmentations_per_image=args.num_augmentations_per_image,
		num_iterations_for_val=args.num_iterations_for_val,
		num_iterations_for_test=args.num_iterations_for_test,
		clustering_alpha=args.clustering_alpha,
		device='cpu'
	)
    
	wandb.init(
		project="genetic-augmentation-optimization",
		config={
			"num_generations": args.num_generations,
			"sol_per_pop": args.sol_per_pop,
			"num_parents_mating": args.num_parents_mating,
			"keep_elitism": args.keep_elitism,
			"keep_parents": args.keep_parents,
			"mutation_percent": args.mutation_percent,
			"dataset": ga_helper.dataset_name,
			"num_ways": ga_helper.num_ways,
			"num_shots": ga_helper.num_shots,
			"subset": ga_helper.subset,
			"model_type": ga_helper.model_type.value,
			"tree_depth": ga_helper.tree_depth,
			"num_augmentations_per_image": ga_helper.num_augmentations_per_image,
			"num_iterations_for_val": ga_helper.num_iterations_for_val,
			"num_iterations_for_test": ga_helper.num_iterations_for_test,
			"seed": args.seed,
			"one_shot_training_loss": args.one_shot_training_loss,
			"one_shot_clustering": args.one_shot_clustering,
			"clustering_alpha": args.clustering_alpha
		}
	)

	ga_instance = pygad.GA(
		num_generations=args.num_generations,
		num_parents_mating=args.num_parents_mating,
		fitness_func=lambda ga_instance, genome, solution_idx: (
			ga_helper.fitness_func_one_shot_training_loss(genome, args.num_shots) if args.one_shot_training_loss
			else ga_helper.fitness_func_one_shot_clustering(genome, args.num_shots) if args.one_shot_clustering
			else ga_helper.fitness_func(genome, args.num_shots)
		),
		initial_population=initial_population(args.sol_per_pop, ga_helper.tree_depth),
		keep_elitism=args.keep_elitism,
		keep_parents=args.keep_parents,
		gene_space=gene_space(ga_helper.tree_depth),
		mutation_percent_genes=args.mutation_percent,
		on_generation=lambda ga_instance: ga_helper.on_generation(ga_instance),
		on_stop=lambda ga_instance, last_gen_fitness_values: ga_helper.on_stop(ga_instance),
		save_solutions=True
	)

	# Run the GA
	ga_instance.run()

	wandb.finish()