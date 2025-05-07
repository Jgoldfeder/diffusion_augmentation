import os

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from dataset_manager import get_dataset_from_torch
from genetic_algorithm import genome_to_tree


dataset_name = 'oxford-iiit-pet'
tree_genome = [0,0.34453242726627215,5,0.4100117273476477,1,0.6908916682199031]
num_augs_per_image = 2

OUTPUT_DIR = f'./full_shot_datasets/{dataset_name}'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

full_dataset, label_to_class = get_dataset_from_torch(dataset_name)
num_classes = len(label_to_class)
print(label_to_class)

node = genome_to_tree(tree_genome)

def gen_aug_image(image, class_name):
        return node.generate_augmentation(image, class_name)

def save_augmented_images(image, label, index, output_subdir):
    label_name = label_to_class[label]
    label_folder = os.path.join(output_subdir, f'{label}_{label_name}')
    os.makedirs(label_folder, exist_ok=True)

    image.save(os.path.join(label_folder, f'{index}.png'))

    for i in range(num_augs_per_image):
        aug_image = gen_aug_image(image, label_name)
        aug_filename = f'{index}_aug_{i}.png'
        aug_image.save(os.path.join(label_folder, aug_filename))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("Processing training data with augmentations...")
skip_to = 296
for i, (image, label) in tqdm(enumerate(train_dataset[skip_to:], start=skip_to), total=len(train_dataset) - skip_to):
    save_augmented_images(image, label, i, TRAIN_DIR)

# Process validation data without augmentation (just original images)
print("Processing test (validation) data...")
for i, (image, label) in tqdm(enumerate(val_dataset), total=len(val_dataset)):
    label_name = label_to_class[label]
    label_folder = os.path.join(TEST_DIR, f'{label}_{label_name}')
    os.makedirs(label_folder, exist_ok=True)
    image.save(os.path.join(label_folder, f'{i}.png'))