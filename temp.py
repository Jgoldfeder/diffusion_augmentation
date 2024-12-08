import os
from torchvision.datasets import SUN397
from PIL import Image
from collections import defaultdict

# Initialize the dataset (Ensure you have already downloaded it)
dataset = SUN397(root='./torch', download=True)

# Create a dictionary to group image paths by class
class_to_images = defaultdict(list)
max_dimensions_by_class = {}

all_classes = dataset.classes
print(f"Number of classes: {len(all_classes)}")

# Iterate through the dataset and group images by class
for class_idx in all_classes:
    first_letter = class_idx[0]
    data_path = os.path.join(os.getcwd(), dataset.root, "SUN397", first_letter, class_idx)
    print(f"<LOG> Data path: {data_path}")
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        class_to_images[class_idx].append(img_path)

# Iterate through each class and find the image with the largest dimension
for class_name, image_paths in class_to_images.items():
    largest_dimension = 0
    largest_image_dimensions = None

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                max_dimension = max(width, height)

                if max_dimension > largest_dimension:
                    largest_dimension = max_dimension
                    largest_image_dimensions = (width, height)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    max_dimensions_by_class[class_name] = largest_image_dimensions
    print(f"Class: {class_name}, Largest Image Dimensions: {largest_image_dimensions}")

# Breakpoint for inspecting the dictionary
import pdb; pdb.set_trace()
