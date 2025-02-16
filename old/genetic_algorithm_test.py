import pygad
import random
import numpy as np
import wandb
import argparse
import time
import os

import AugmentationNode
from AugmentationNode import print_tree
import fitness_score
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

from fitness_score import create_datasets
import get_base_model
from make_augmenations_from_tree import generate_augmentations_from_tree
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch

from CustomDataset import FewShotDataset

seed = -1
dataset = ''
num_shots = -1
tree_depth = -1
sol_per_pop = -1
# NOTE these whill be set later by parser args

fitness_cache = dict()

segment_aug_manager = SegmentAugmentationManager()
color_aug_manager = ColorControlNetAugmentationManager()
canny_aug_manager = CannyAugmentationManager()
nerf_aug_manager = NerfAugmentationManager()
depth_aug_manager = DepthAugmentationManager()
aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset, val_dataset, test_dataset = None, None, None


def tree_to_string(node, level=0, direction='root'):
    tree_str = ''
    if node:
        if not node.left and not node.right:
            # Leaf nodes: only show augmentation type
            edge_info = f"(Augmentation: {node.augmentation_type})"
        else:
            # Non-leaf nodes: show augmentation type and probabilities
            edge_info = f"(Augmentation: {node.augmentation_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
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

def compute_test_accuracy(augmentation_tree, device):
    print('[LOG] Computing test accuracy on combined train and val datasets')
    print('For both classical augs and the best aug tree')

    dataset_path = f"few_shot_datasets/{dataset}/{num_shots}_shot/seed_{seed}"
    train_dataset = FewShotDataset(dataset_path, dataset_type='train')
    test_dataset = FewShotDataset(dataset_path, dataset_type='test')

    # classical_dataset = ClassicalDataset(train_dataset, transform, duplicate_factor=6)
    augmented_dataset = generate_augmentations_from_tree(augmentation_tree, train_dataset, aug_managers, transform)

    output_dir = 'sample_tree_augmentations'
    for idx, (img, label, class_name) in enumerate(augmented_dataset):
        # Generate the filename using the specified format
        file_name = f"{class_name}_{label}_{idx}.png"  # Use .png or desired image format
        file_path = os.path.join(output_dir, file_name)
        
        # Convert the tensor image to a PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # remove normalization
            img = transforms.ToPILImage()(img)  # Convert to PIL Image
        
        # Save the image to the specified folder
        img.save(file_path)


    model = get_base_model.get_resnet50(num_outputs=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(400):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels, class_names in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels, class_names in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = epoch_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{400}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')

        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "epoch": epoch
        })

        if epoch == 399:  # On the last epoch
            # Calculate confusion matrix
            confusion_matrix = torch.zeros(5, 5)  # Assuming 5 classes
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels, class_names in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            print("\nConfusion Matrix:")
            print(confusion_matrix)
            
            # Log confusion matrix to wandb
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=[str(i) for i in range(5)]  # Replace with actual class names if available
                )
            })

    return test_accuracy

num_times_fitness_calculated = 0
def fitness_function(ga_instance, augmentation_tree_genome, solution_idx):
    """Calculates the fitness of an individual."""
    global fitness_cache
    genome_number = genome_to_number(augmentation_tree_genome)
    if genome_number in fitness_cache:
        print('fitness function call using cached fitness')
        return fitness_cache[genome_number]

    augmentation_tree = genome_to_tree(augmentation_tree_genome)
    # loss = fitness_score.fitness_score(augmentation_tree, train_dataset, val_dataset, aug_managers)
    loss = random.random()
    fitness = -1 * loss

    print('fitness function called')
    print_tree(augmentation_tree)
    print('fitness:', fitness)
    print('Time since start (seconds):', int(time.time() - start_time))

    global num_times_fitness_calculated
    num_times_fitness_calculated += 1

    fitness_cache[genome_number] = fitness
    return fitness

def gene_space():
    """Defines the gene space for the GA."""
    gene_space = []
    for i in range(2 ** tree_depth - 1):
        gene_space.extend([[i for i in range(len(AugmentationNode.augmentation_types))], {"low": 0.3, "high": 0.7}])
    return gene_space

def initial_population():
    population = []
    augmentation_cycler = 0
    num_genes = 2 * (2 ** tree_depth - 1)
    added_baseline = False
    for i in range(sol_per_pop):
        genome = []
        for j in range(0, num_genes, 2):
            aug_type = random.randint(0, len(AugmentationNode.augmentation_types) - 1)
            left_prob = random.uniform(0.3, 0.7)
            genome.extend([aug_type, left_prob])
        genome[0] = augmentation_cycler

        # have one tree that is our baseline
        if not added_baseline and AugmentationNode.augmentation_types[augmentation_cycler] == 'classical':
            added_baseline = True
            for j in range(2, num_genes, 2):
                genome[j] = AugmentationNode.augmentation_types.index('none')

        augmentation_cycler += 1
        augmentation_cycler %= len(AugmentationNode.augmentation_types)
        population.append(genome)
    # print(population)
    return population

# Initialize GA
fitness_progress = []  # To store fitness values for each generation

num_generations_finished = 0
def on_generation(ga_instance):
    global num_times_fitness_calculated, num_generations_finished

    print('Finished evolution generation')
    print('Num times fitness function called:', num_times_fitness_calculated)

    num_generations_finished += 1
    num_times_fitness_calculated = 0

    best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_tree = genome_to_tree(best_solution[0])
    best_fitness = best_solution[1]

    print(f'Best tree for generation {num_generations_finished}:')
    print_tree(best_tree)
    fitness_progress.append(best_fitness)
    print('Time since start (seconds):', int(time.time() - start_time))

    folder_name = "genetic_runs"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{dataset}_seed_{seed}.txt"
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "a") as file:
        file.write('Best tree:')
        file.write(str(best_solution[0]) + '\n')
        file.write(str(best_fitness) + '\n')
        file.write(tree_to_string(best_tree) + "\n")
        file.write(f'{num_generations_finished} generations finished\n')
        file.write('\n\n')

    # Log metrics to wandb
    wandb.log({
        "generation": num_generations_finished,
        "best_fitness": best_fitness,
        "population_fitness_mean": np.mean(ga_instance.last_generation_fitness),
        "population_fitness_std": np.std(ga_instance.last_generation_fitness)
    })

def on_stop(ga_instance, last_gen_fitness_values):
    best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_tree = genome_to_tree(best_solution[0])
    best_tree_accuracy = compute_test_accuracy(best_tree, "cuda")
    print('Best tree accuracy:', best_tree_accuracy)
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
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots (examples per class)')
    parser.add_argument('--tree_depth', type=int, required=True, help='Depth of the augmentation tree')
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
            "tree_depth": args.tree_depth,
            "dataset": args.dataset,
            "seed": args.seed,
            "num_shots": args.num_shots
        }
    )

    random.seed(args.seed)

    global sol_per_pop, tree_depth, seed, dataset, num_shots, train_dataset, val_dataset, test_dataset

    dataset = args.dataset
    seed = args.seed
    num_shots = args.num_shots
    tree_depth = args.tree_depth
    sol_per_pop = args.sol_per_pop

    train_dataset, val_dataset, test_dataset = create_datasets(dataset, seed, num_shots)

    num_genes = 2 * (2 ** tree_depth - 1)

    ga_instance = pygad.GA(
        num_generations=args.num_generations,
        num_parents_mating=args.num_parents_mating,
        fitness_func=fitness_function,
        # sol_per_pop=sol_per_pop,
        # num_genes=num_genes,
        initial_population=initial_population(),
        keep_elitism=args.keep_elitism,
        keep_parents=args.keep_parents,
        gene_space=gene_space(),
        mutation_percent_genes=args.mutation_percent,
        on_generation=on_generation,
        on_stop=on_stop,
        save_solutions=True
    )

    # Run the GA
    ga_instance.run()

    wandb.finish()

def string_to_tree(tree_string):
    print(tree_string)
    # Split the string into lines and remove empty lines
    lines = [line for line in tree_string.strip().split('\n') if line.strip()]
    
    # Create a map of (level, position) -> line for easy lookup
    tree_map = {}
    for line in lines:
        # Calculate level based on indentation (2 spaces per level)
        level = (len(line) - len(line.lstrip())) // 2
        # Extract position (root, L, R, etc.)
        position = line.lstrip().split(':')[0].strip()
        tree_map[(level, position)] = line.strip()

    def create_node(level, position):
        if (level, position) not in tree_map:
            return None

        line = tree_map[(level, position)]
        # Extract augmentation type from between parentheses
        aug_info = line[line.find("(")+1:line.find(")")].strip()
        aug_type = aug_info.split('Augmentation:')[1].split(',')[0].strip()
        
        node = AugmentationNode.AugmentationNode(aug_type)
        
        # Extract probabilities if they exist (non-leaf nodes)
        if 'L_prob:' in line:
            left_prob = float(line.split('L_prob:')[1].split(',')[0].strip())
            right_prob = float(line.split('R_prob:')[1].split(')')[0].strip())
            node.left_child_probability = left_prob
            node.right_child_probability = right_prob
            
            # Create children
            if position == 'root':
                node.left = create_node(level + 1, 'L')
                node.right = create_node(level + 1, 'R')
            else:
                node.left = create_node(level + 1, position + 'L')
                node.right = create_node(level + 1, position + 'R')
        
        return node

    # Start creating the tree from the root
    root = create_node(0, 'root')
    return root

if __name__ == "__main__":
    main()
    # compute_test_accuracy(AugmentationNode.initialize_augmentation_tree(tree_depth), 'cuda')
