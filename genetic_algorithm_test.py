import pygad
import random
import numpy as np

import AugmentationNode
import fitness_score

# Problem parameters
tree_depth = 3

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
    root_node.right_child_probability = genome[2]
    queue = [root_node]
    for i in range(3, len(genome), 6):
        node = queue.pop(0)
        new_node_left = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i])])
        new_node_left.left_child_probability = genome[i + 1]
        new_node_left.right_child_probability = genome[i + 2]
        new_node_right = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i + 3])])
        new_node_right.left_child_probability = genome[i + 4]
        new_node_right.right_child_probability = genome[i + 5]
        node.left = new_node_left
        node.right = new_node_right
        queue.append(node.left)
        queue.append(node.right)
    return root_node

def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    augmentation_tree = genome_to_tree(augmentation_tree_genome)
    fitness = fitness_score.fitness_score(augmentation_tree)
    return fitness

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for i in range(2 ** tree_depth - 1):
        gene_space.extend([[i for i in range(len(AugmentationNode.augmentation_types))], {"low": 0.0, "high": 1.0}, {"low": 0.0, "high": 1.0}])
    return gene_space

# GA parameters
num_generations = 2
num_parents_mating = 2
sol_per_pop = 5
num_genes = 3 * (2 ** tree_depth - 1)

# Initialize GA
fitness_progress = []  # To store fitness values for each generation

def on_generation(ga_instance):
    """Callback executed at the end of each generation."""
    fitness_progress.append(ga_instance.best_solution()[1])  # Save best fitness of generation

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
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

print(f"Fitness progression saved to {fitness_file}")

# Output the results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: {solution}")
print(f"Fitness: {solution_fitness}")