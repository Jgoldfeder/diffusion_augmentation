num_times_fitness_called = 0
import pygad
import random
import numpy as np

import AugmentationNode
import fitness_score
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

# Problem parameters
tree_depth = 4

segment_aug_manager = SegmentAugmentationManager()
color_aug_manager = ColorControlNetAugmentationManager()
canny_aug_manager = CannyAugmentationManager()
nerf_aug_manager = NerfAugmentationManager()
depth_aug_manager = DepthAugmentationManager()
aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

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


def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    augmentation_tree = genome_to_tree(augmentation_tree_genome)
    loss = fitness_score.fitness_score(augmentation_tree, aug_managers)
    fitness = -1 * loss
    print_tree(augmentation_tree)
    print('fitness function called')
    return fitness

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for i in range(2 ** tree_depth - 1):
        gene_space.extend([[i for i in range(len(AugmentationNode.augmentation_types))], {"low": 0.0, "high": 1.0}])
    return gene_space

# GA parameters
# TODO make sure that num generations * sol_per_pop is the number of times fitness function is called
num_generations = 5
num_parents_mating = 20
keep_elitism = 0
sol_per_pop = 30
num_genes = 2 * (2 ** tree_depth - 1)

# Initialize GA
fitness_progress = []  # To store fitness values for each generation

def on_generation(ga_instance):
    """Callback executed at the end of each generation."""
    global num_times_fitness_called
    print('Finished evolution generation')
    print('Num times fitness function called:', num_times_fitness_called)
    num_times_fitness_called = 0
    fitness_progress.append(ga_instance.best_solution()[1])  # Save best fitness of generation

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    keep_elitism=keep_elitism,
    num_genes=num_genes,
    gene_space=gene_space(),
    mutation_percent_genes=10,
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

# print(f"Fitness progression saved to {fitness_file}")

# # Output the results
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print(f"Best solution: {solution}")
# print_tree(genome_to_tree(solution))
# print(f"Fitness: {solution_fitness}")