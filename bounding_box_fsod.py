import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# -------------------------------------------------------------
# CONFIGURE PATHS
# -------------------------------------------------------------
DATASET_ROOT = "./torch/fsod"   # change this!
IMG_DIRS = [
    os.path.join(DATASET_ROOT, "part_1"),
    os.path.join(DATASET_ROOT, "part_2"),
]
# print(IMG_DIRS)

TRAIN_JSON = os.path.join(DATASET_ROOT, "annotations", "fsod_train.json")

OUTPUT_DIR = os.path.join('./', "output_fsod")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# LOAD ANNOTATIONS
# -------------------------------------------------------------
with open(TRAIN_JSON, "r") as f:
    ann_data = json.load(f)

images = ann_data["images"]
annotations = ann_data["annotations"]

# Build image ID → annotations list
ann_by_img = {}
for ann in annotations:
    img_id = ann["image_id"]
    ann_by_img.setdefault(img_id, []).append(ann)

# -------------------------------------------------------------
# SEARCH FOR IMAGE FILE ACROSS part_1 + part_2
# -------------------------------------------------------------
def find_image_file(filename):
	path = os.path.join(DATASET_ROOT, filename)
	if os.path.exists(path):
		return path
	return None

# -------------------------------------------------------------
# SAVE IMAGE WITH BOUNDING BOXES TO DISK
# -------------------------------------------------------------
def save_image_with_boxes(img_info, save_path):
    img_path = find_image_file(img_info["file_name"])
    if img_path is None:
        print("Image not found:", img_info["file_name"])
        return
    
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    ax = plt.gca()

    # Draw bounding boxes
    for ann in ann_by_img.get(img_info["id"], []):
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    
    plt.title(img_info["file_name"])
    plt.axis("off")

    # Save to file
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", save_path)

# -------------------------------------------------------------
# PICK 5 RANDOM IMAGES AND SAVE THEM
# -------------------------------------------------------------
sample_imgs = random.sample(images, 5)

for img_info in sample_imgs:
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(img_info['file_name'])[0]}_boxed.jpg"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_with_boxes(img_info, output_path)
