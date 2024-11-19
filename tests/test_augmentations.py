import unittest
from PIL import Image
from pathlib import Path
import random
import torch
import time
import sys 
import os

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager
from augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from augmentation_models.NerfAugmentation import NerfAugmentationManager

sample_image_paths = [
    "/home/vaibhav/diffusion_augmentation/torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg",
    "/home/vaibhav/diffusion_augmentation/torch/caltech256/256_ObjectCategories/002.american-flag/002_0001.jpg",
]

output_dir = "/home/vaibhav/diffusion_augmentation/test_augmentations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

controlnet_manager = ControlNetAugmentationManager()
canny_augmented, depth_augmented, segmentation_augmented = controlnet_manager.generate_augmentations(sample_image_paths)
for image_path, augmented_image in canny_augmented.items():
    class_name = Path(image_path).stem
    augmented_image.save(f"{output_dir}/canny_{class_name}_augmented.png")

for image_path, augmented_image in depth_augmented.items():
    class_name = Path(image_path).stem  
    augmented_image.save(f"{output_dir}/depth_{class_name}_augmented.png")

for image_path, augmented_image in segmentation_augmented.items():
    class_name = Path(image_path).stem
    augmented_image.save(f"{output_dir}/segmentation_{class_name}_augmented.png")

color_controlnet_manager = ColorControlNetAugmentationManager()
color_augmented = color_controlnet_manager.generate_augmentations(sample_image_paths)

for image_path, augmented_image in color_augmented.items():
    class_name = Path(image_path).stem
    augmented_image.save(f"{output_dir}/color_{class_name}_augmented.png")

nerf_manager = NerfAugmentationManager()
nerf_augmented = nerf_manager.generate_augmentations(sample_image_paths)
for image_path, augmented_image in nerf_augmented.items():
    class_name = Path(image_path).stem
    augmented_image.save(f"{output_dir}/nerf_{class_name}_augmented.png")

print("Augmented images have been saved")