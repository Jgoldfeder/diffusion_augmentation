import AugmentationNode
# from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
# from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
# from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
# from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
# from image_augmentation_models.DepthAugmentation import DepthAugmentationManager
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

def generate_augmentations_from_tree(root: AugmentationNode, dataset, class_to_label_map, aug_managers) -> list:
    augmentations = []
    labels = []

    segment_aug_manager = aug_managers[0]
    color_aug_manager = aug_managers[1]
    canny_aug_manager = aug_managers[2]
    nerf_aug_manager = aug_managers[3]
    depth_aug_manager = aug_managers[4]

    for entry in dataset:
        #run through the tree 5 times and compose augmentations based on the given tree
        curr_image = entry[0]
        label = entry[1]
        class_name = class_to_label_map[label]
        #print(class_to_label_map)
        augmentations.append(curr_image)
        labels.append(label)

        for i in range(5):
            print(f"Generating augmentation {i+1} for class {class_name}")
            #start from the root and traverse the tree down randomly based on the left and right probabilities
            curr_node = root
            while curr_node.left and curr_node.right:
                if random.random() < curr_node.left_child_probability:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right

                if curr_node.parent_edge_type == "segment":
                    curr_image = segment_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.parent_edge_type == "color":
                    curr_image = color_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.parent_edge_type == "canny":
                    curr_image = canny_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.parent_edge_type == "nerf":
                    curr_image = nerf_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.parent_edge_type == "depth":
                    curr_image = depth_aug_manager.generate_augmentations([curr_image], [class_name])[0]
                elif curr_node.parent_edge_type == "classical":
                    curr_image = classical_aug_transform(curr_image)
                elif curr_node.parent_edge_type == "none":
                    curr_image = curr_image # do nothing explicitly

            #add the final image to the list
            augmentations.append(curr_image)
            labels.append(label)

    # Convert lists to a list of tuples (image, class) for DataLoader compatibility
    print(f"Augmentations: {augmentations}")
    print(f"Labels: {labels}")
    combined_dataset = list(zip(augmentations, labels))
    
    return combined_dataset


def test_make_augmentations():
    #initialize all the augmentation managers
    segment_aug_manager = SegmentAugmentationManager()
    color_aug_manager = ColorControlNetAugmentationManager()
    canny_aug_manager = CannyAugmentationManager()
    nerf_aug_manager = NerfAugmentationManager()
    depth_aug_manager = DepthAugmentationManager()

    aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

    tree = AugmentationNode.initialize_augmentation_tree()
    print_tree(tree)

    #create a sample dataset
    sample_dataset = [(Image.open("torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg"), 0)]
    class_to_label_map = {0: "ak47"}

    augmented_dataset = generate_augmentations_from_tree(tree, sample_dataset, class_to_label_map, aug_managers)

    #save all the images to a folder
    import os
    if not os.path.exists("test_augmented_images"):
        os.makedirs("test_augmented_images")
    for i, (image, label) in enumerate(augmented_dataset):

        image.save(f"test_augmented_images/augmented_{i}.jpg")

    print(f"Saved {len(augmented_dataset)} augmented images to augmented_images/")

if __name__ == '__main__':
    test_generateda_augmentations()
   

    