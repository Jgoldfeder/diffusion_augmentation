import pygad
import random
import numpy as np
import wandb
import argparse
import time

import AugmentationNode
import fitness_score
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

from fitness_score import create_datasets

segment_aug_manager = SegmentAugmentationManager()
color_aug_manager = ColorControlNetAugmentationManager()
canny_aug_manager = CannyAugmentationManager()
nerf_aug_manager = NerfAugmentationManager()
depth_aug_manager = DepthAugmentationManager()
aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

# Problem parameters
tree_depth = 4


def print_tree(node, level=0, direction='root'):
    if node:
        if direction == 'root':
            edge_info = f"(root, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        else:
            edge_info = f"(edge: {node.parent_edge_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        print('  ' * level + f"{direction}: {edge_info}")
        if node.left:
            print_tree(node.left, level + 1, 'L')
        if node.right:
            print_tree(node.right, level + 1, 'R')

def genome_to_tree(genome):
    root_node = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[0])])
    root_node.left_child_probability = genome[1]
    root_node.right_child_probability = 1 - genome[1]
    queue = [root_node]
    for i in range(2, len(genome), 4):
        node = queue.pop(0)
        new_node_left = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i])])
        new_node_left.left_child_probability = genome[i + 1]
        new_node_left.right_child_probability = 1 - genome[i + 1]
        new_node_right = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i + 2])])
        new_node_right.left_child_probability = genome[i + 3]
        new_node_right.right_child_probability = 1 - genome[i + 3]
        node.left = new_node_left
        node.right = new_node_right
        queue.append(node.left)
        queue.append(node.right)
    return root_node

start_time = int(time.time())
train_dataset, val_dataset, test_dataset, label_to_class = create_datasets()

num_times_fitness_called = 0
def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    augmentation_tree = genome_to_tree(augmentation_tree_genome)
    loss = fitness_score.fitness_score(augmentation_tree, train_dataset, val_dataset, aug_managers, label_to_class)
    fitness = -1 * loss

    print('fitness function called')
    print_tree(augmentation_tree)
    print('fitness:', fitness)
    print('Time since start (seconds):', int(time.time() - start_time))

    global num_times_fitness_called
    num_times_fitness_called += 1

    return fitness

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for i in range(2 ** tree_depth - 1):
        gene_space.extend([[i for i in range(len(AugmentationNode.augmentation_types))], {"low": 0.3, "high": 0.7}])
    return gene_space

# GA parameters
# TODO make sure that num generations * sol_per_pop is the number of times fitness function is called
num_generations = 3
num_parents_mating = 4
keep_elitism = 1
keep_parents = 4
sol_per_pop = 10
num_genes = 2 * (2 ** tree_depth - 1)

# Initialize GA
fitness_progress = []  # To store fitness values for each generation

num_generations_finished = 0
def on_generation(ga_instance):
    global num_times_fitness_called, num_generations_finished

    print('Finished evolution generation')
    print('Num times fitness function called:', num_times_fitness_called)

    num_generations_finished += 1
    num_times_fitness_called = 0

    best_solution = ga_instance.best_solution()
    best_tree = genome_to_tree(best_solution[0])
    best_fitness = best_solution[1]

    print(f'Best tree for generation {num_generations_finished}:')
    print_tree(best_tree)
    fitness_progress.append(best_fitness)
    print('Time since start (seconds):', int(time.time() - start_time))

    # Log metrics to wandb
    wandb.log({
        "generation": num_generations_finished,
        "best_fitness": best_fitness,
        "population_fitness_mean": np.mean(ga_instance.last_generation_fitness),
        "population_fitness_std": np.std(ga_instance.last_generation_fitness)
    })

def parse_args():
    parser = argparse.ArgumentParser(description='Run genetic algorithm for augmentation tree optimization')
    parser.add_argument('--num_generations', type=int, default=3, help='Number of generations')
    parser.add_argument('--sol_per_pop', type=int, default=10, help='Solutions per population')
    parser.add_argument('--num_parents_mating', type=int, default=4, help='Number of parents for mating')
    parser.add_argument('--keep_elitism', type=int, default=1, help='Number of elites to keep')
    parser.add_argument('--keep_parents', type=int, default=4, help='Number of parents to keep')
    parser.add_argument('--mutation_percent', type=int, default=10, help='Mutation percentage')
    return parser.parse_args()

def main():
    args = parse_args()
    
    wandb.init(
        project="genetic-augmentation-optimization",
        config={
            "num_generations": args.num_generations,
            "sol_per_pop": args.sol_per_pop,
            "num_parents_mating": args.num_parents_mating,
            "keep_elitism": args.keep_elitism,
            "keep_parents": args.keep_parents,
            "mutation_percent": args.mutation_percent,
            "tree_depth": tree_depth
        }
    )

    ga_instance = pygad.GA(
        num_generations=args.num_generations,
        num_parents_mating=args.num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=args.sol_per_pop,
        keep_elitism=args.keep_elitism,
        keep_parents=args.keep_parents,
        num_genes=num_genes,
        gene_space=gene_space(),
        mutation_percent_genes=args.mutation_percent,
        on_generation=on_generation
    )

    # Run the GA
    ga_instance.run()

    # Save fitness progression to a file
    fitness_file = "fitness_progression.csv"
    with open(fitness_file, "w") as file:
        file.write("Generation,Fitness\n")
        for gen, fitness in enumerate(fitness_progress):
            file.write(f"{gen},{fitness}\n")

    wandb.finish()

if __name__ == "__main__":
    main()