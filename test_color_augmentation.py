from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from PIL import Image
import os

color_controlnet_manager = ColorControlNetAugmentationManager()

# Load the input image
image_path = "few_shot_datasets/flowers102/2_shot/seed_48/train/4_columbine/0.png"
input_image = Image.open(image_path)

augmented_image = color_controlnet_manager.generate_augmentations([input_image], ["columbine"])[0]

os.makedirs("test_augmentation_outputs", exist_ok=True)
output_path = "test_augmentation_outputs/augmented_columbine.png"
augmented_image.save(output_path)

print(f"Augmented image saved to: {output_path}")
