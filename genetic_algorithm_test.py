import pygad
import random
import numpy as np

# Problem parameters
target_string = "aaaa"
letters = ['x', 'b', 'c', 'd']
tree_depth = 4  # Number of nodes in the tree

def tree_to_string(individual):
    """
    Converts a tree represented by an individual into a string.
    The tree is encoded as a linear sequence of [char, prob1, prob2, ...].
    """
    string = ""
    current_node = 0  # Start at the root node

    for _ in range(tree_depth):
        # Get the character at the current node
        char_index = current_node * 3
        char = letters[int(individual[char_index])]

        # Get the probabilities
        prob1 = individual[char_index + 1]
        prob2 = individual[char_index + 2]

        # Add the character to the output string
        string += char

        # Determine the next node based on the probabilities
        if prob1 > prob2:
            next_node = (current_node + 1) % tree_depth
        else:
            next_node = (current_node + 2) % tree_depth

        current_node = next_node

    return string

def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    # instantiate augmentation tree from genome
    # create dataset from augmentation tree
    # train NN on dataset for a few epochs
    # get the associated loss value and assign this as fitness
    generated_string = tree_to_string(augmentation_tree_genome)
    # Fitness is higher the closer the generated string is to the target
    fitness = -sum(abs(ord(g) - ord(t)) for g, t in zip(generated_string, target_string))
    return fitness

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for i in range(tree_depth):
        gene_space.extend([{"low": 0.0, "high": 1.0}, {"low": 0.0, "high": 1.0}, [0, 1, 2, 3]])
    return gene_space

# GA parameters
num_generations = 2
num_parents_mating = 4
sol_per_pop = 10
num_genes = tree_depth * 3  # 3 elements (char, float1, float2) per node

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
print(f"Generated string: {tree_to_string(solution)}")