from dataset_manager import get_classical_transform
import os
from PIL import Image

def augment_one_shot_datasets():
    datasets = ["stanford_cars", "flowers102", "caltech256", "stanford_dogs", "food101"]

    for dataset in datasets:
        base_path = os.path.join("few_shot_datasets", dataset, "5_ways", "1_shot")
        for subset in range(41, 50):
            train_path = os.path.join(base_path, f"subset_{subset}", "train")
            for class_name in os.listdir(train_path):
                full_data_path = os.path.join(train_path, class_name)
                #load 0.png
                img = Image.open(os.path.join(full_data_path, "0.png"))
                transform = get_classical_transform()
                augmented_img = transform(img)
                augmented_img.save(os.path.join(full_data_path, "1.png"))

                print(f"Augmented image for {full_data_path}")


def remove_augmented_images():
    datasets = ["stanford_cars", "flowers102", "caltech256", "stanford_dogs", "food101"]

    for dataset in datasets:
        base_path = os.path.join("few_shot_datasets", dataset, "5_ways", "1_shot")
        for subset in range(41, 50):
            train_path = os.path.join(base_path, f"subset_{subset}", "train")
            for class_name in os.listdir(train_path):
                full_data_path = os.path.join(train_path, class_name)
                os.remove(os.path.join(full_data_path, "1.png"))

if __name__ == "__main__":
    remove_augmented_images()