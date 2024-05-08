import os
import random

# Constants
FOCAL_LENGTH = 512.0

def read_index_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def generate_paths(line):
    image_path, base_name = line.split(' ')
    dir_path, image_file = os.path.split(image_path)
    # Path for depth images
    depth_path = dir_path.replace('color_left', 'depth') + '/' + image_file.replace('.JPEG', '.PNG')
    # Variations are stored in the control_depth_variations directory
    variations_dir = os.path.join(os.getcwd(), 'control_depth_variations', '0')
    variations = [f"{base_name}_{i}.png" for i in range(2)] + [f"{base_name}_original.png"]
    valid_paths = []
    for var in variations:
        var_full_path = os.path.join(variations_dir, var)
        if os.path.exists(var_full_path):
            valid_paths.append((var_full_path, depth_path))
    return valid_paths

def write_dataset(filename, dataset):
    with open(filename, 'w') as file:
        for image_path, depth_path in dataset:
            file.write(f"{image_path} {depth_path} {FOCAL_LENGTH}\n")

def main():
    index_file = 'augmented_files.txt'
    training_file = 'train_aug.txt'
    testing_file = 'test_aug.txt'
    
    lines = read_index_file(index_file)
    dataset = []
    for line in lines:
        dataset.extend(generate_paths(line))
    print(dataset)
    
    random.shuffle(dataset)
    split_point = int(0.9 * len(dataset))
    
    training_dataset = dataset[:split_point]
    testing_dataset = dataset[split_point:]
    
    write_dataset(training_file, training_dataset)
    write_dataset(testing_file, testing_dataset)
    print(f"Dataset split into {len(training_dataset)} training and {len(testing_dataset)} testing entries.")

if __name__ == "__main__":
    main()