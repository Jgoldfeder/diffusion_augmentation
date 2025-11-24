import argparse
import json
import os
import shutil
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
from torchvision.transforms import (
    Compose,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)
from torchvision.transforms import functional as F


def get_classical_augmentations():
    """Returns classical augmentation pipeline."""
    return Compose([
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10)
    ])


def transform_bbox(bbox, img_size, transform_params):
    """
    Transform bounding box coordinates based on applied augmentations.
    
    bbox: [x, y, w, h] in COCO format
    img_size: (width, height) of original image
    transform_params: dict with keys 'hflip', 'vflip', 'rotation'
    
    Returns: transformed [x, y, w, h]
    """
    import math
    import numpy as np
    
    x, y, w, h = bbox
    width, height = img_size
    
    # Convert to corner format
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Apply horizontal flip
    if transform_params.get('hflip', False):
        x1_new = width - x2
        x2_new = width - x1
        x1, x2 = x1_new, x2_new
    
    # Apply vertical flip
    if transform_params.get('vflip', False):
        y1_new = height - y2
        y2_new = height - y1
        y1, y2 = y1_new, y2_new
    
    # Apply rotation
    angle = transform_params.get('rotation', 0)
    if abs(angle) > 0.01:  # Only process if there's meaningful rotation
        # Get all four corners of the bounding box
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        # Center of image
        cx, cy = width / 2, height / 2
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotate each corner around image center
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Translate to origin
            tx = corner_x - cx
            ty = corner_y - cy
            
            # Rotate
            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a
            
            # Translate back
            rotated_corners.append([rx + cx, ry + cy])
        
        rotated_corners = np.array(rotated_corners)
        
        # Get axis-aligned bounding box that contains all rotated corners
        x1 = rotated_corners[:, 0].min()
        y1 = rotated_corners[:, 1].min()
        x2 = rotated_corners[:, 0].max()
        y2 = rotated_corners[:, 1].max()
        
        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
    
    # Convert back to COCO format
    new_x = x1
    new_y = y1
    new_w = x2 - x1
    new_h = y2 - y1
    
    # Ensure positive width and height
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    return [new_x, new_y, new_w, new_h]


def apply_augmentation_with_tracking(img, transform):
    """
    Apply augmentation and track which transforms were applied.
    Returns: (transformed_img, transform_params)
    """
    # Manually apply each transform to track what happened
    transform_params = {}
    
    # ColorJitter doesn't affect bboxes
    if hasattr(transform.transforms[0], 'brightness'):
        img = transform.transforms[0](img)
    
    # Horizontal flip
    if torch.rand(1) < 0.5:
        img = F.hflip(img)
        transform_params['hflip'] = True
    else:
        transform_params['hflip'] = False
    
    # Vertical flip
    if torch.rand(1) < 0.5:
        img = F.vflip(img)
        transform_params['vflip'] = True
    else:
        transform_params['vflip'] = False
    
    # Rotation (simplified - just apply the transform)
    # For production, you'd want to track rotation angle too
    angle = (torch.rand(1).item() - 0.5) * 20  # -10 to +10 degrees
    img = F.rotate(img, angle)
    transform_params['rotation'] = angle
    
    return img, transform_params


def visualize_annotations(img, annotations, categories_dict, output_path):
    """
    Draw bounding boxes on image and save visualization.
    
    Args:
        img: PIL Image
        annotations: list of annotation dicts with 'bbox' and 'category_id'
        categories_dict: dict mapping category_id -> category info (with 'name')
        output_path: where to save the visualization
    """
    # Create a copy to draw on
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Try to load a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each bounding box
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Get category name
        cat_id = ann["category_id"]
        cat_name = categories_dict.get(cat_id, {}).get("name", f"id_{cat_id}")
        
        # Draw label with background
        text = cat_name
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox, fill="yellow")
        draw.text((x1, y1), text, fill="black", font=font)
    
    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis_img.save(output_path)


def augment_dataset(subset_path, base_output_root, num_augmentations=2):
    """
    Create augmented version of dataset.
    
    Args:
        subset_path: relative path like "subset_0"
        base_output_root: base directory (e.g., "few_shot_datasets/bounding_boxes/fsod")
        num_augmentations: number of augmented copies per training image
    """
    # Original dataset path
    original_root = os.path.join(base_output_root, subset_path)
    
    # New augmented dataset path
    augmented_root = os.path.join(base_output_root, "classical", subset_path)
    
    print(f"Original dataset: {original_root}")
    print(f"Augmented dataset: {augmented_root}")
    
    # Create output directories
    os.makedirs(augmented_root, exist_ok=True)
    
    # Process train split
    train_input_dir = os.path.join(original_root, "train")
    train_output_dir = os.path.join(augmented_root, "train")
    vis_output_dir = os.path.join(augmented_root, "vis")
    
    if not os.path.exists(train_input_dir):
        raise FileNotFoundError(f"Train directory not found at {train_input_dir}")
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(os.path.join(train_output_dir, "images"), exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Load train annotations
    train_ann_file = os.path.join(train_input_dir, "annotations.json")
    with open(train_ann_file, "r") as f:
        train_coco = json.load(f)
    
    # Build categories dict for visualization
    categories_dict = {cat["id"]: cat for cat in train_coco["categories"]}
    
    # Get augmentation transform
    aug_transform = get_classical_augmentations()
    
    # Process training images and annotations
    new_images = []
    new_annotations = []
    
    next_image_id = max([img["id"] for img in train_coco["images"]]) + 1
    next_ann_id = max([ann["id"] for ann in train_coco["annotations"]]) + 1
    
    # Build annotation lookup
    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in train_coco["annotations"]:
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)
    
    # Add original images and annotations
    new_images.extend(train_coco["images"])
    new_annotations.extend(train_coco["annotations"])
    
    # Copy original images (handling nested directory structures)
    for img_info in train_coco["images"]:
        src_path = os.path.join(train_input_dir, "images", img_info["file_name"])
        dst_path = os.path.join(train_output_dir, "images", img_info["file_name"])
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        # Create visualization for original image
        img = Image.open(src_path).convert("RGB")
        img_annotations = ann_by_image.get(img_info["id"], [])
        vis_path = os.path.join(vis_output_dir, img_info["file_name"])
        visualize_annotations(img, img_annotations, categories_dict, vis_path)
    
    print(f"Processing {len(train_coco['images'])} training images...")
    
    # Create augmented copies
    for img_info in train_coco["images"]:
        original_img_id = img_info["id"]
        original_file_name = img_info["file_name"]
        img_path = os.path.join(train_input_dir, "images", original_file_name)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        
        # Get annotations for this image
        img_annotations = ann_by_image.get(original_img_id, [])
        
        # Create N augmented versions
        for aug_idx in range(num_augmentations):
            # Apply augmentation with tracking
            aug_img, transform_params = apply_augmentation_with_tracking(
                img, aug_transform
            )
            
            # Create new image entry
            file_name_base, file_ext = os.path.splitext(original_file_name)
            new_file_name = f"{file_name_base}_aug{aug_idx}{file_ext}"
            
            new_img_info = {
                "id": next_image_id,
                "file_name": new_file_name,
                "width": img_width,
                "height": img_height,
            }
            new_images.append(new_img_info)
            
            # Save augmented image
            aug_img_path = os.path.join(train_output_dir, "images", new_file_name)
            os.makedirs(os.path.dirname(aug_img_path), exist_ok=True)
            aug_img.save(aug_img_path)
            
            # Transform and add annotations
            transformed_annotations = []
            for ann in img_annotations:
                new_bbox = transform_bbox(
                    ann["bbox"],
                    (img_width, img_height),
                    transform_params
                )
                
                new_ann = {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": ann["category_id"],
                    "bbox": new_bbox,
                    "area": new_bbox[2] * new_bbox[3],
                    "iscrowd": ann.get("iscrowd", 0),
                }
                new_annotations.append(new_ann)
                transformed_annotations.append(new_ann)
                next_ann_id += 1
            
            # Create visualization for augmented image
            vis_path = os.path.join(vis_output_dir, new_file_name)
            visualize_annotations(aug_img, transformed_annotations, categories_dict, vis_path)
            
            next_image_id += 1
        
        if (original_img_id % 10 == 0):
            print(f"  Processed {original_img_id}/{len(train_coco['images'])} images")
    
    # Save augmented train annotations
    augmented_train_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": train_coco["categories"],
    }
    
    augmented_train_ann_file = os.path.join(train_output_dir, "annotations.json")
    with open(augmented_train_ann_file, "w") as f:
        json.dump(augmented_train_coco, f, indent=2)
    
    print(f"Original training images: {len(train_coco['images'])}")
    print(f"Augmented training images: {len(new_images)}")
    print(f"Original annotations: {len(train_coco['annotations'])}")
    print(f"Augmented annotations: {len(new_annotations)}")
    print(f"Visualizations saved to: {vis_output_dir}")
    
    # Copy test split without augmentation
    test_input_dir = os.path.join(original_root, "test")
    test_output_dir = os.path.join(augmented_root, "test")
    
    if os.path.exists(test_input_dir):
        print("Copying test split (no augmentation)...")
        shutil.copytree(test_input_dir, test_output_dir, dirs_exist_ok=True)
    else:
        print("Warning: No test directory found, skipping test split")
    
    print(f"\nAugmented dataset created at: {augmented_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply classical augmentations to FSOD few-shot dataset."
    )
    parser.add_argument(
        "--subset-path",
        type=str,
        required=True,
        help="Relative path to the subset directory (e.g., 'subset_0').",
    )
    parser.add_argument(
        "--base-output-root",
        type=str,
        default="few_shot_datasets/bounding_boxes/fsod",
        help="Base directory containing subset folders.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=2,
        help="Number of augmented copies to create per training image.",
    )
    args = parser.parse_args()
    
    augment_dataset(
        subset_path=args.subset_path,
        base_output_root=args.base_output_root,
        num_augmentations=args.num_augmentations,
    )


if __name__ == "__main__":
    main()