import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch import nn

import get_base_model
from CustomDataset import split_train_test, split_train_val, ValDataset, TreeAugmentedDataset, FewShotDataset, split_into_two
from make_augmenations_from_tree import generate_augmentations_from_tree
from AugmentationNode import initialize_augmentation_tree, print_tree
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

device = "cpu"

def create_datasets(dataset, seed, num_shots):
    dataset_path = f"few_shot_datasets/{dataset}/{num_shots}_shot/seed_{seed}"
    train_dataset = FewShotDataset(dataset_path, dataset_type='train')
    test_dataset = FewShotDataset(dataset_path, dataset_type='test')

    fold_0_dataset, fold_1_dataset = split_into_two(train_dataset)

    return fold_0_dataset, fold_1_dataset, test_dataset

def get_val_loss(model, train_loader, val_loader, optimizer, criterion, device):
    for epoch in range(20):
        for batch in train_loader:
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    return val_loss

def get_fold_loss_for_tree(my_tree, train_dataset, val_dataset, aug_managers) -> float:
    model = get_base_model.get_resnet50(num_outputs=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    augmented_dataset = generate_augmentations_from_tree(my_tree, train_dataset, aug_managers, transform)
    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    val_dataset_with_transform = ValDataset(val_dataset, transform)
    val_loader = DataLoader(val_dataset_with_transform, batch_size=32, shuffle=False)

    return get_val_loss(model, augmented_loader, val_loader, optimizer, criterion, device)


def fitness_score(my_tree, train_dataset, val_dataset, aug_managers) -> float:
    val_loss_0 = get_fold_loss_for_tree(my_tree, train_dataset, val_dataset, aug_managers)
    val_loss_1 = get_fold_loss_for_tree(my_tree, val_dataset, train_dataset, aug_managers)
    return (val_loss_0 + val_loss_1) / 2

if __name__ == '__main__':
    #initialize some random tree and run the fitness score
    individual = initialize_augmentation_tree(depth=4)
    aug_managers = [SegmentAugmentationManager(), ColorControlNetAugmentationManager(), CannyAugmentationManager(), NerfAugmentationManager(), DepthAugmentationManager()]
    train_dataset, val_dataset, test_dataset, new_label_to_class = create_datasets(42)
    for i in range(5):
        print(fitness_score(individual, train_dataset, val_dataset, aug_managers, new_label_to_class))
