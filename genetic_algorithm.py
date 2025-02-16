import random

import wandb
import pygad
import logging
import argparse
import numpy as np

import network_model
import dataset_manager
from network_model import ModelResults, ModelType
from dataset_manager import FolderDataset
from augmentation_tree import BinaryAugmentationNode, AugmentationType, TreeAugmentedDataset, ProbabilityLimits

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
		num_iterations_for_val: int, num_iterations_for_test: int, device
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
		self.device = device

		self.train_path = dataset_manager.get_dataset_path(self.dataset_name, self.num_ways, self.num_shots, self.subset, train=True)
		self.test_path = dataset_manager.get_dataset_path(self.dataset_name, self.num_ways, self.num_shots, self.subset, train=False)

		self.tree_evals_per_generation = [0]
		self.fitness_cache: dict[float, float] = dict()

	def fitness_func(self, genome):
		genome_number = genome_to_number(genome)
		if genome_number in self.fitness_cache:
			logging.info('fitness function call using cached fitness')
			return self.fitness_cache[genome_number]
		self.tree_evals_per_generation[-1] += 1

		node = genome_to_tree(genome)
		dataset = TreeAugmentedDataset(self.train_path, node, self.num_augmentations_per_image)
		train_dataset, val_dataset = dataset_manager.split_train_val(dataset)
		model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
		model_results: ModelResults = network_model.train_and_test(model, train_dataset, val_dataset, self.num_iterations_for_val, self.device)

		fitness = -1 * model_results.losses[-1]
		self.fitness_cache[genome_number] = fitness
		return fitness

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
			"population_fitness_std": np.std(ga_instance.last_generation_fitness)
		})

	def on_stop(self, ga_instance):
		best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
		best_tree = genome_to_tree(best_solution[0])

		train_dataset = TreeAugmentedDataset(self.train_path, best_tree, self.num_augmentations_per_image)
		test_dataset = FolderDataset(self.test_path)

		model = network_model.get_model_for_finetune(self.model_type, self.num_ways)
		model_results: ModelResults = network_model.train_and_test(model, train_dataset, test_dataset, self.num_iterations_for_test, self.device)
		best_tree_accuracy = model_results.accs[-1]

		logging.info(f'Best tree accuracy: {best_tree_accuracy}')

		for i in range(self.num_iterations_for_test):
			wandb.log({
				'train_loss': model_results.train_losses[i],
				'train_acc': model_results.train_accs[i],
				'test_loss': model_results.losses[i],
				'test_acc': model_results.accs[i],
				'epoch': i
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
    return parser.parse_args()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	args = parse_args()

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
			"num_iterations_for_test": ga_helper.num_iterations_for_test,
			"seed": args.seed,
		}
	)

	ga_instance = pygad.GA(
		num_generations=args.num_generations,
		num_parents_mating=args.num_parents_mating,
		fitness_func=lambda ga_instance, genome, solution_idx: ga_helper.fitness_func(genome),
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