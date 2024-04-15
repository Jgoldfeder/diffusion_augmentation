import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class FewShotDataset(Dataset):
    def __init__(self, root_dir, num_unique_files=0, include_variations=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_unique_files (int): Number of unique files to include in the dataset.
            include_variations (bool): Whether to include variations or just the original images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_id in sorted(os.listdir(root_dir)):  # Ensure class order is consistent
            class_path = os.path.join(root_dir, class_id)
            originals = [fname for fname in os.listdir(class_path) if fname.endswith("_original.png")]
            # Sort to ensure consistency in selection
            originals.sort()

            selected_originals = originals[:num_unique_files] if num_unique_files > 0 else originals

            # Build a base name list from selected originals to ensure matching variations are found
            base_names = ["_".join(fname.split("_")[:-1]) for fname in selected_originals]

            for base_name in base_names:
                # Always include the original image
                original_path = os.path.join(class_path, f"{base_name}_original.png")
                self.image_paths.append(original_path)
                self.labels.append(class_id)

                if include_variations:
                    # Include any variations of the selected originals
                    for variation in ['0', '1']:
                        variation_fname = f"{base_name}_{variation}.png"
                        if variation_fname in os.listdir(class_path):
                            self.image_paths.append(os.path.join(class_path, variation_fname))
                            self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # print(img_path)
        label = int(self.labels[idx])  # Assuming class_id can be directly converted

        if self.transform:
            image = self.transform(image)

        return image, label
    

# Check if the two datasets have the same original images
if __name__ == "__main__":
    no_variations = FewShotDataset(root_dir="./control_augmented_images", num_unique_files=5, include_variations=False)
    #print the number of classes and images
    print(len(no_variations))
    #display the image
    print(no_variations[0])

    variations = FewShotDataset(root_dir="./control_augmented_images", num_unique_files=5, include_variations=True)
    #print the number of classes and images
    print(len(variations))
    print(variations[0])

    # Assuming you've already initialized `no_variations` and `variations` datasets

    # Extract img_paths from both datasets
    no_variations_paths = [no_variations[i][2] for i in range(len(no_variations))]
    variations_paths = [variations[i][2] for i in range(len(variations))]

    # Extract just the original image paths
    no_variations_originals = set([path for path in no_variations_paths if path.endswith("_original.png")])
    variations_originals = set([path for path in variations_paths if path.endswith("_original.png")])

    # sort the originals list
    no_variations_originals = sorted(no_variations_originals)
    variations_originals = sorted(variations_originals)

    # loop through them and check if each r the same
    for i in range(len(no_variations_originals)):
        if no_variations_originals[i] != variations_originals[i]:
            print("not the same")
            break
    # Compare
    # Results
    print(f"Original images in variations: {len(variations_originals)}")
    print(f"Original images in no_variations: {len(no_variations_originals)}")

