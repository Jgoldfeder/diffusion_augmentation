import pygad
import numpy as np
import time
import random

import AugmentationNode
from AugmentationNode import print_tree

tree_depth = 2
num_genes = 2 * (2 ** tree_depth - 1)
sol_per_pop = 12

total_occurances = [0 for i in range(len(AugmentationNode.augmentation_types))]
all_genomes_from_fitness = []
genome_numbers = set()
genome_repeat_count = 0
fitness_cache = dict()

def tree_to_string(node, level=0, direction='root'):
    tree_str = ''
    if node:
        if direction == 'root':
            edge_info = f"(root, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        else:
            edge_info = f"(edge: {node.augmentation_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        tree_str += '  ' * level + f"{direction}: {edge_info}" + '\n'
        if node.left:
            tree_str += tree_to_string(node.left, level + 1, 'L')
        if node.right:
            tree_str += tree_to_string(node.right, level + 1, 'R')
    return tree_str

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

def genome_to_number(genome):
    number = 0
    mult = 1
    for i in range(0, len(genome), 2):
        number += (genome[i] + genome[i+1]) * mult
        mult *= 10
    return number

def string_to_genome(tree_string):
    # Dictionary to map augmentation names to indices
    aug_type_to_index = {
        'canny': 0,
        'depth': 1, 
        'seg': 2,
        'color': 3,
        'nerf': 4,
        'classical': 5,
        'none': 6
    }
    lines = [line.strip() for line in tree_string.strip().split('\n') if line.strip()]
    tree_map = {}
    for line in lines:
        level = (len(line) - len(line.lstrip())) // 2
        position = line.lstrip().split(':')[0]
        tree_map[(level, position)] = line

    genome = []
    max_level = max(level for level, _ in tree_map.keys())
    
    root_line = tree_map[(0, 'root')]
    left_prob = float(root_line.split('L_prob:')[1].split(',')[0].strip())
    genome.extend([aug_type_to_index['none'], left_prob])
    
    current_positions = ['root']
    for level in range(max_level):
        next_positions = []
        for pos in current_positions:
            if pos == 'root':
                left_pos = 'L'
                right_pos = 'R'
            else:
                left_pos = pos + 'L'
                right_pos = pos + 'R'
                
            # Get left child
            if (level + 1, left_pos) in tree_map:
                left_line = tree_map[(level + 1, left_pos)]
                aug_type = left_line.split('edge:')[1].split(',')[0].strip()
                left_prob = float(left_line.split('L_prob:')[1].split(',')[0].strip())
                genome.extend([aug_type_to_index[aug_type], left_prob])
                next_positions.append(left_pos)
            
            # Get right child
            if (level + 1, right_pos) in tree_map:
                right_line = tree_map[(level + 1, right_pos)]
                aug_type = right_line.split('edge:')[1].split(',')[0].strip()
                left_prob = float(right_line.split('L_prob:')[1].split(',')[0].strip())
                genome.extend([aug_type_to_index[aug_type], left_prob])
                next_positions.append(right_pos)
        
        current_positions = next_positions
    
    return genome

start_time = int(time.time())

num_times_fitness_called = 0
def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    global all_genomes_from_fitness, genome_numbers, genome_repeat_count, fitness_cache
    genome_number = genome_to_number(augmentation_tree_genome)
    if genome_number in fitness_cache:
        return fitness_cache[genome_number]
    all_genomes_from_fitness.append(augmentation_tree_genome)
    if genome_number in genome_numbers:
        genome_repeat_count += 1
    genome_numbers.add(genome_number)

    augmentation_tree = genome_to_tree(augmentation_tree_genome)
    k = AugmentationNode.augmentation_types.index('color')
    fitness = 0
    for genome_part in augmentation_tree_genome:
        if genome_part == k:
            fitness += 1
    fitness += random.random() / 2

    # print('fitness function called')
    # print_tree(augmentation_tree)
    # print('fitness:', fitness)
    # print('Time since start (seconds):', int(time.time() - start_time))

    global num_times_fitness_called
    num_times_fitness_called += 1

    fitness_cache[genome_number] = fitness
    return fitness

def initial_population():
    population = []
    augmentation_cycler = 0
    for i in range(sol_per_pop):
        genome = []
        for j in range(0, num_genes, 2):
            aug_type = random.randint(0, len(AugmentationNode.augmentation_types) - 1)
            left_prob = random.uniform(0.3, 0.7)
            genome.extend([aug_type, left_prob])
        genome[0] = augmentation_cycler
        augmentation_cycler += 1
        augmentation_cycler %= len(AugmentationNode.augmentation_types)
        population.append(genome)
    # print(population)
    return population

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for j in range(0, num_genes, 2):
        gene_space.extend([[i for i in range(len(AugmentationNode.augmentation_types))], {"low": 0.3, "high": 0.7}])
    return gene_space

# Initialize GA
fitness_progress = []  # To store fitness values for each generation

num_generations_finished = 0
def on_generation(ga_instance):
    global num_times_fitness_called, num_generations_finished

    # print('Finished evolution generation')
    # print('Num times fitness function called:', num_times_fitness_called)

    num_generations_finished += 1
    num_times_fitness_called = 0

    # print(ga_instance.population)
    global total_occurances
    for ga_member in ga_instance.population:
        aug_type_as_int = int(ga_member[0])
        total_occurances[aug_type_as_int] += 1
    # print(total_occurances)

    # best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    # best_tree = genome_to_tree(best_solution[0])
    # best_fitness = best_solution[1]

    # print(f'Best tree for generation {num_generations_finished}:')
    # print_tree(best_tree)
    # fitness_progress.append(best_fitness)
    # print('Time since start (seconds):', int(time.time() - start_time))

def main():
    global total_occurances, all_genomes_from_fitness, genome_numbers, genome_repeat_count

    totals = []
    genome_repeat_counts = []

    for i in range(1000):
        total_occurances = [0 for i in range(len(AugmentationNode.augmentation_types))]
        all_genomes_from_fitness = []
        genome_numbers = set()
        genome_repeat_count = 0


        ga_instance = pygad.GA(
            num_generations=10,
            num_parents_mating=6,
            fitness_func=fitness_function,
            # sol_per_pop=sol_per_pop,
            # num_genes=num_genes,
            gene_space=gene_space(),
            initial_population=initial_population(),
            keep_elitism=1,
            keep_parents=1,
            mutation_percent_genes=10,
            on_generation=on_generation,
            save_solutions=True
        )

        ga_instance.run()

        totals.append(total_occurances)
        genome_repeat_counts.append(genome_repeat_count)

    # here we find the avg val of aug type at index 3
    curr_sum = 0
    for row in totals:
        curr_sum += row[3]
    print('avg num appearances of aug type 3 at tree node 0:', curr_sum / len(totals))
    print('min num appearances of aug type 3 at tree node 0:', min([totals[i][3] for i in range(len(totals))]))

    print()

    print('num repeat fitness calls:')
    print(min(genome_repeat_counts))
    print(sum(genome_repeat_counts) / len(genome_repeat_counts))
    print(max(genome_repeat_counts))


if __name__ == "__main__":
    main()