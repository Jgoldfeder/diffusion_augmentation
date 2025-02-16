import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from CustomDataset import FewShotDataset
from make_augmenations_from_tree import generate_augmentations_from_tree
import get_base_model
import AugmentationNode

from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

from fitness_score import fitness_score, create_datasets

segment_aug_manager = SegmentAugmentationManager()
color_aug_manager = ColorControlNetAugmentationManager()
canny_aug_manager = CannyAugmentationManager()
nerf_aug_manager = NerfAugmentationManager()
depth_aug_manager = DepthAugmentationManager()
aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

tree_depth = 2
dataset = 'flowers102'
seed = 50
num_shots = 2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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

def compute_fitness_score(augmentation_tree, device):
    train_dataset, val_dataset, test_dataset = create_datasets(dataset, seed, num_shots)
    loss = fitness_score(augmentation_tree, train_dataset, val_dataset, aug_managers)
    fitness = loss * -1
    return fitness

def print_tree(node, level=0, direction='root'):
    if node:
        if direction == 'root':
            edge_info = f"(root, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        else:
            edge_info = f"(edge: {node.augmentation_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        print('  ' * level + f"{direction}: {edge_info}")
        if node.left:
            print_tree(node.left, level + 1, 'L')
        if node.right:
            print_tree(node.right, level + 1, 'R')

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

def compute_random_tree_accuracy():
	compute_test_accuracy(AugmentationNode.initialize_augmentation_tree(tree_depth), 'cuda')

if __name__ == '__main__':
    my_tree = AugmentationNode.initialize_augmentation_tree(tree_depth)
    print(tree_to_string(my_tree))
    my_tree.augmentation_type = 'color'
    my_tree.left.augmentation_type = 'none'
    my_tree.right.augmentation_type = 'none'
    print(compute_fitness_score(my_tree, 'cuda'))
