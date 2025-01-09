import torch
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
from make_augmenations_from_tree import generate_augmentations_from_tree
from AugmentationNode import AugmentationNode
from AugmentationNode import initialize_augmentation_tree

def create_sample_tree():
    # Create a simple tree with different augmentation types
    root = AugmentationNode(left_child_probability=0.6)
    
    # Left branch
    root.left = AugmentationNode(parent_edge_type="classical", left_child_probability=0.5)
    root.left.left = AugmentationNode(parent_edge_type="color")
    root.left.right = AugmentationNode(parent_edge_type="canny")
    
    # Right branch
    root.right = AugmentationNode(parent_edge_type="segment", left_child_probability=0.7)
    root.right.left = AugmentationNode(parent_edge_type="depth")
    root.right.right = AugmentationNode(parent_edge_type="nerf")
    
    return root

def visualize_augmentations(original_images, augmented_images):
    num_images = len(augmented_images)
    plt.figure(figsize=(15, 5))
    
    # Plot original images
    for idx, img in enumerate(original_images):
        plt.subplot(2, 6, idx + 1)
        plt.imshow(img)
        plt.title(f'Original {idx+1}')
        plt.axis('off')
    
    # Plot some augmented images
    for idx, img in enumerate(augmented_images[:10]):  # Show first 10 augmentations
        plt.subplot(2, 6, idx + 7)
        plt.imshow(img)
        plt.title(f'Aug {idx+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    sample_paths = ["torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg",
                     "torch/caltech256/256_ObjectCategories/001.ak47/001_0002.jpg"]
    sample_images = [Image.open(path) for path in sample_paths]

    # Create augmentation tree
    aug_tree = initialize_augmentation_tree(depth=4)

    # Generate augmentations
    augmented_images = generate_augmentations_from_tree(aug_tree, sample_images, ["ak47", "ak47"])
    
    # Visualize results
    visualize_augmentations(sample_images, augmented_images)
    
    # Print some statistics
    print(f"Number of original images: {len(sample_images)}")
    print(f"Number of augmented images: {len(augmented_images)}")
    print(f"Expected number of augmentations: {len(sample_images) * (1 + 5)}")  # Original + 5 augmentations per image

if __name__ == "__main__":
    main()