import torchvision.transforms as transforms
import argparse

from CustomDataset import FewShotDataset, ClassicalDataset

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch import nn
import torch
import os
import random

import get_base_model
import AugmentationNode
from AugmentationNode import print_tree

from make_augmenations_from_tree import generate_augmentations_from_tree

from fitness_score import create_datasets

from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

# Move these variable declarations after the global statement
global dataset, seed, num_shots, train_dataset, val_dataset, test_dataset
dataset = None
num_shots = -1
seed = -1
train_dataset, val_dataset, test_dataset = None, None, None

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

    return test_accuracy

if __name__ == '__main__':
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Few-shot learning training script')
    parser.add_argument('--seed', type=int, default=41, help='Random seed')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset name')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Update dataset paths to use arguments
    base_path = f'few_shot_datasets/{args.dataset}/{args.shots}_shot/seed_{args.seed}'
    train_dataset = FewShotDataset(base_path, dataset_type='train')
    test_dataset = FewShotDataset(base_path, dataset_type='test')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classical_dataset = ClassicalDataset(train_dataset, transform, duplicate_factor=6)

    print(len(train_dataset))
    print(train_dataset[0])

    train_loader = DataLoader(classical_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #train the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, args.ways)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(400):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels, class_names in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
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


    # Remove the global statement from here since it's now at the top
    dataset = args.dataset
    seed = args.seed
    num_shots = args.shots

    train_dataset, val_dataset, test_dataset = create_datasets(dataset, seed, num_shots)


    sample_tree_string = '''root: (Augmentation: classical, L_prob: 0.38, R_prob: 0.62)
  L: (Augmentation: none, L_prob: 0.30, R_prob: 0.70)
    L: (Augmentation: none)
    R: (Augmentation: none)
  R: (Augmentation: none, L_prob: 0.43, R_prob: 0.57)
    L: (Augmentation: none)
    R: (Augmentation: none)'''

    augmentation_tree = string_to_tree(sample_tree_string)

    #compute test accuracy
    test_accuracy = compute_test_accuracy(augmentation_tree, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
