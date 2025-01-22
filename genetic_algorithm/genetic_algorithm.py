import numpy as np
from typing import List, Tuple, Callable
import random
from AugmentationNode import initialize_augmentation_tree, print_tree
import AugmentationNode

class GeneticAlgorithm:
    def __init__( 
        self,
        tree_depth: int,
        num_aug_types: int,
        fitness_func: Callable,
        population_size: int = 10,
        generations: int = 10,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
    ):
        self.tree_depth = tree_depth
        self.num_nodes = 2 ** (tree_depth - 1)
        self.genome_size = 2 * self.num_nodes
        
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.best_fitnesses = []
        self.best_genomes = []

        self.num_aug_types = num_aug_types
        self.min_probability = 0.3
        self.max_probability = 0.7

    def initialize_population(self) -> List[np.ndarray]:
        """Create initial random population with alternating integers and probabilities"""
        population = []
        for _ in range(self.population_size):
            genome = np.zeros(self.genome_size)
            # Alternate between integers (0-6) and probabilities (0.3-0.7)
            for i in range(self.genome_size):
                if i % 2 == 0:  # Even indices for integers
                    genome[i] = np.random.randint(0, self.num_aug_types)
                else:  # Odd indices for probabilities
                    genome[i] = np.random.uniform(self.min_probability, self.max_probability)
            population.append(genome)
        return population

    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Calculate fitness for each individual"""
        return [self.fitness_func(individual) for individual in population]

    def select_parent(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """Tournament selection"""
        # NOTE might want to change this implementation to just use the top num_parents individuals
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover"""
        mask = np.random.rand(self.genome_size) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Mutate genome with different strategies for alternating integers and probabilities"""
        mutated = genome.copy()
        
        for i in range(self.genome_size):
            if np.random.random() < self.mutation_rate:
                if i % 2 == 0:  # Even indices for integers
                    mutated[i] = np.random.randint(0, self.num_aug_types)
                else:  # Odd indices for probabilities
                    # NOTE do we want gaussian nudging or uniform random selection 
                    # delta = np.random.normal(0, 0.1)  # Small random change
                    # mutated[i] = np.clip(mutated[i] + delta, 0.3, 0.7)
                    mutated[i] = np.random.uniform(self.min_probability, self.max_probability)
        
        return mutated

    def evolve(self) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm"""
        population = self.initialize_population()

        for gen in range(self.generations):
            # Evaluate current population
            fitnesses = self.evaluate_population(population)
            
            # Keep track of best solution
            current_best_idx = np.argmax(fitnesses)
            self.best_fitnesses.append(fitnesses[current_best_idx])
            self.best_genomes.append(population[current_best_idx])

            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.elite_size):
                new_population.append(population[sorted_indices[i]])

            # Fill rest of population with crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                # TODO put crossover back into this step
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:self.population_size]

            print(f"End of generation {gen + 1}")
            print(f"Best Genome = {self.best_genomes[-1]}")
            print(f"Best Fitness = {self.best_fitnesses[-1]}")

        return self.best_genomes[-1], self.best_fitnesses[-1]


def tree_to_genome(node):
    if node is None:
        return []
    
    genome = []
    queue = [node]
    
    while queue:
        current = queue.pop(0)
        
        # Handle edge type encoding - special case for root node
        if current.parent_edge_type is None:
            genome.append(-1)  # Use -1 to represent None/root node
        else:
            aug_type_index = AugmentationNode.augmentation_types.index(current.parent_edge_type)
            genome.append(aug_type_index)
        
        # Add probability distributions
        genome.append(current.left_child_probability)
        
        # Add children to queue for BFS traversal
        if current.left is not None:
            queue.append(current.left)
        if current.right is not None:
            queue.append(current.right)
    
    return genome

def genome_to_tree(genome):
    root_node = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[0])])
    root_node.left_child_probability = genome[1]
    root_node.right_child_probability = 1 - genome[1]
    queue = [root_node]
    for i in range(2, len(genome), 4):
        node = queue.pop(0)
        new_node_left = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i])])
        new_node_left.left_child_probability = genome[i + 1]
        new_node_left.right_child_probability = 0 if genome[i + 1] == 0 else 1 - genome[i + 1]
        new_node_right = AugmentationNode.AugmentationNode(AugmentationNode.augmentation_types[int(genome[i + 2])])
        new_node_right.left_child_probability = genome[i + 3]
        new_node_right.right_child_probability = 0 if genome[i + 3] == 0 else 1 - genome[i + 3]
        node.left = new_node_left
        node.right = new_node_right
        queue.append(node.left)
        queue.append(node.right)
    return root_node


# Example usage:
def example_fitness_function(genome: np.ndarray) -> float:
    return np.sum(genome)

def test_tree_genome_conversion():
    tree = initialize_augmentation_tree()
    genome = tree_to_genome(tree)
    print(genome)
    reconstructed_tree = genome_to_tree(genome)
    print_tree(reconstructed_tree)



if __name__ == "__main__":
    ga_instance = GeneticAlgorithm(
        tree_depth=3,
        num_aug_types=100,
        fitness_func=example_fitness_function,
        population_size=100,
        generations=100,
        mutation_rate=0.1,
        elite_size=10
    )

    ga_instance.evolve()
