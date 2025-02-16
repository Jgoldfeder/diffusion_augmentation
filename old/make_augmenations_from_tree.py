import AugmentationNode
import torchvision.transforms as transforms
import random
from PIL import Image
import torch

from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

from AugmentationNode import print_tree
from CustomDataset import TreeAugmentedDataset

classical_aug_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)), # NOTE should this resize be here? Otherwise errors on some classes, ex. kangaroo-101 there is an image of size (300, 182)
            transforms.RandomCrop(size=(224, 224)),  # Randomly crop to 224x224 pixels
            transforms.ColorJitter(
                brightness=0.4,  # Adjust brightness (factor range [0.6, 1.4])
                contrast=0.4,    # Adjust contrast (factor range [0.6, 1.4])
                saturation=0.4,  # Adjust saturation (factor range [0.6, 1.4])
                hue=0.2          # Adjust hue (factor range [-0.2, 0.2])
            ),
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            transforms.RandomVerticalFlip(p=0.5),    # Random vertical flip with 50% probability
            transforms.RandomRotation(degrees=30)   # Random rotation within [-30, 30] degrees
        ])

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def generate_augmentations_from_tree(root: AugmentationNode, dataset, aug_managers, transform) -> list:
    augmentations = []
    labels = []
    class_names = []
    segment_aug_manager = aug_managers[0]
    color_aug_manager = aug_managers[1]
    canny_aug_manager = aug_managers[2]
    nerf_aug_manager = aug_managers[3]
    depth_aug_manager = aug_managers[4]

    for image, label, class_name in dataset:
        #run through the tree 5 times and compose augmentations based on the given tree
        augmentations.append(image)
        labels.append(label)
        class_names.append(class_name)

        for i in range(5):
            print(f"Generating augmentation {i+1} for class {class_name}")
            #start from the root and traverse the tree down randomly based on the left and right probabilities
            curr_node = root
            curr_image = image
            while curr_node:
                print(f"Current node: {curr_node.augmentation_type}")
                if curr_node.augmentation_type == "seg":
                    curr_image = segment_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.augmentation_type == "color":
                    curr_image = color_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.augmentation_type == "canny":
                    curr_image = canny_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.augmentation_type == "nerf":
                    curr_image = nerf_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.augmentation_type == "depth":
                    curr_image = depth_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.augmentation_type == "classical":
                    curr_image = classical_aug_transform(curr_image)
                elif curr_node.augmentation_type == "none":
                    curr_image = curr_image # do nothing explicitly
                
                #traverse the tree down randomly based on the left and right probabilities
                if random.random() < curr_node.left_child_probability:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right

            #add the final image to the list
            augmentations.append(curr_image)
            labels.append(label)
            class_names.append(class_name)

    #return a list of tuples (image, label, class_name)
    dataset = TreeAugmentedDataset(list(zip(augmentations, labels, class_names)), transforms.Compose([classical_aug_transform, transform]))
    return dataset


def test_make_augmentations():
    #initialize all the augmentation managers
    segment_aug_manager = SegmentAugmentationManager()
    color_aug_manager = ColorControlNetAugmentationManager()
    canny_aug_manager = CannyAugmentationManager()
    nerf_aug_manager = NerfAugmentationManager()
    depth_aug_manager = DepthAugmentationManager()

    aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]
    tree = AugmentationNode.initialize_augmentation_tree(depth=2)
    print_tree(tree)

    #create a sample dataset
    sample_dataset = [(Image.open("torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg"), 0, "ak47")]

    augmented_dataset = generate_augmentations_from_tree(tree, sample_dataset, aug_managers, transform)

    #save all the images to a folder
    import os
    if not os.path.exists("test_augmented_images"):
        os.makedirs("test_augmented_images")
    for i, (image, label, class_name) in enumerate(augmented_dataset):
        if isinstance(image, torch.Tensor):
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # remove normalization
            image = transforms.ToPILImage()(image)
        image.save(f"test_augmented_images/augmented_{i}.jpg")

    print(f"Saved {len(augmented_dataset)} augmented images to augmented_images/")

if __name__ == '__main__':
    test_make_augmentations()
   

    