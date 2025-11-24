import wandb
import random
import time

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    ColorJitter,
)
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -----------------------------
# Augmentation tree imports
# -----------------------------
from augmentation_tree import BinaryAugmentationNode, ProbabilityLimits
from image_augmentation_models.augmentation_manager import AugmentationType


# ============================================================
# Classical augmentation + bbox tracking utilities
# ============================================================

def get_classical_augmentations():
    """Returns classical augmentation pipeline.

    We only really use the ColorJitter part; flips/rotation are applied
    manually so we can track them.
    """
    return Compose([
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
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


def apply_classical_augmentation_with_tracking(img, color_jitter):
    """
    Apply classical augmentation and track which transforms were applied.
    Returns: (transformed_img, transform_params)
    """
    transform_params = {}
    
    # ColorJitter doesn't affect bboxes
    if color_jitter is not None:
        img = color_jitter(img)
    
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
    
    # Rotation
    angle = (torch.rand(1).item() - 0.5) * 20  # -10 to +10 degrees
    img = F.rotate(img, angle)
    transform_params['rotation'] = angle
    
    return img, transform_params


def create_augmentation_tree_no_classical(num_levels):
    """
    Create a random augmentation tree that excludes CLASSICAL augmentations
    but allows NONE nodes.
    
    Args:
        num_levels: depth of the tree
    
    Returns:
        BinaryAugmentationNode tree
    """
    import random

    def make_tree_recursive(levels_remaining):
        node = BinaryAugmentationNode()
        
        # Allow NONE, disallow CLASSICAL
        valid_types = [t for t in AugmentationType if t is not AugmentationType.CLASSICAL]
        
        if not valid_types:
            raise ValueError("No valid augmentation types available (after excluding CLASSICAL)")
        
        node.augmentation_type = random.choice(valid_types)
        node.left_probability = ProbabilityLimits.get_random_probability()
        
        if levels_remaining > 1:
            node.left = make_tree_recursive(levels_remaining - 1)
            node.right = make_tree_recursive(levels_remaining - 1)
        
        return node
    
    return make_tree_recursive(num_levels)


# ============================================================
# FSOD datasets
# ============================================================

class PrecomputedAugmentedFSODTrainDataset(Dataset):
    """
    Precompute all augmentations once in memory:
      - For each original train image:
          * store the original sample
          * generate `num_augmentations` augmented versions using:
              - augmentation_tree (no classical nodes, NONE allowed)
              - classical augmentation with tracking
          * store augmented image tensors and transformed targets
    """
    def __init__(
        self,
        split_root: str,
        ann_file: str,
        cat_id_to_idx: Dict[int, int],
        cat_id_to_name: Dict[int, str],
        augmentation_tree: BinaryAugmentationNode,
        num_augmentations: int,
        color_jitter: ColorJitter,
    ):
        self.samples: List[Tuple[torch.Tensor, Dict[str, Any]]] = []

        img_root = os.path.join(split_root, "images")
        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]

        # Build image_id -> list[annotations]
        ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            ann_by_image.setdefault(img_id, []).append(ann)

        # Precompute a representative class name per image
        image_id_to_class_name: Dict[int, str] = {}
        for img_info in images:
            img_id = img_info["id"]
            anns = ann_by_image.get(img_id, [])
            if anns:
                cat_id = anns[0]["category_id"]
                image_id_to_class_name[img_id] = cat_id_to_name.get(cat_id, "unknown")
            else:
                image_id_to_class_name[img_id] = "unknown"

        # Precompute all samples
        print("Precomputing original + augmented training samples in memory...")
        for img_idx, img_info in enumerate(images):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_path = os.path.join(img_root, file_name)

            img = Image.open(img_path).convert("RGB")
            img_width, img_height = img.size
            anns = ann_by_image.get(img_id, [])

            # --- build original sample ---
            boxes = []
            labels = []
            areas = []
            iscrowd = []

            for ann in anns:
                x, y, w, h = ann["bbox"]  # [x, y, w, h]
                x1, y1 = x, y
                x2, y2 = x + w, y + h

                boxes.append([x1, y1, x2, y2])
                labels.append(cat_id_to_idx[ann["category_id"]])
                areas.append(ann.get("area", w * h))
                iscrowd.append(ann.get("iscrowd", 0))

            if boxes:
                boxes_t = torch.tensor(boxes, dtype=torch.float32)
                labels_t = torch.tensor(labels, dtype=torch.int64)
                areas_t = torch.tensor(areas, dtype=torch.float32)
                iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            else:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                areas_t = torch.zeros((0,), dtype=torch.float32)
                iscrowd_t = torch.zeros((0,), dtype=torch.int64)

            image_id_tensor = torch.tensor([img_id])
            target_orig: Dict[str, Any] = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": image_id_tensor,
                "area": areas_t,
                "iscrowd": iscrowd_t,
            }
            img_tensor_orig = F.to_tensor(img)
            self.samples.append((img_tensor_orig, target_orig))

            # --- build augmented samples ---
            class_name = image_id_to_class_name.get(img_id, "unknown")

            for aug_idx in range(num_augmentations):
                # 1) tree augmentation (assumed not to affect geometry)
                tree_aug_img = augmentation_tree.generate_augmentation(img, class_name)

                # 2) classical augmentation (with tracking)
                final_img, transform_params = apply_classical_augmentation_with_tracking(
                    tree_aug_img, color_jitter
                )

                # transform bboxes
                aug_boxes = []
                aug_labels = []
                aug_areas = []
                aug_iscrowd = []

                for ann, lbl, crowd_val in zip(anns, labels, iscrowd):
                    new_bbox = transform_bbox(
                        ann["bbox"],
                        (img_width, img_height),
                        transform_params,
                    )
                    x, y, w, h = new_bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    aug_boxes.append([x1, y1, x2, y2])
                    aug_labels.append(lbl)
                    aug_areas.append(w * h)
                    aug_iscrowd.append(crowd_val)

                if aug_boxes:
                    boxes_t = torch.tensor(aug_boxes, dtype=torch.float32)
                    labels_t = torch.tensor(aug_labels, dtype=torch.int64)
                    areas_t = torch.tensor(aug_areas, dtype=torch.float32)
                    iscrowd_t = torch.tensor(aug_iscrowd, dtype=torch.int64)
                else:
                    boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                    labels_t = torch.zeros((0,), dtype=torch.int64)
                    areas_t = torch.zeros((0,), dtype=torch.float32)
                    iscrowd_t = torch.zeros((0,), dtype=torch.int64)

                # We can reuse original img_id; it doesn't affect training
                image_id_tensor = torch.tensor([img_id])
                target_aug: Dict[str, Any] = {
                    "boxes": boxes_t,
                    "labels": labels_t,
                    "image_id": image_id_tensor,
                    "area": areas_t,
                    "iscrowd": iscrowd_t,
                }
                img_tensor_aug = F.to_tensor(final_img)
                self.samples.append((img_tensor_aug, target_aug))

            if (img_idx + 1) % 10 == 0:
                print(f"  Precomputed {img_idx + 1}/{len(images)} base images")

        print(f"Total precomputed train samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class FSODFewShotTestDataset(Dataset):
    """
    Test dataset: no augmentation, just load images and targets.
    """
    def __init__(
        self,
        split_root: str,
        ann_file: str,
        cat_id_to_idx: Dict[int, int],
    ):
        self.img_root = os.path.join(split_root, "images")
        self.cat_id_to_idx = cat_id_to_idx

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_root, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.ann_by_image.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        image_id_tensor = torch.tensor([img_id])
        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id_tensor,
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        img_tensor = F.to_tensor(img)
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# Category mapping, model, train/eval
# ============================================================

def build_category_mapping(train_ann_file: str):
    with open(train_ann_file, "r") as f:
        coco = json.load(f)

    categories = coco["categories"]
    orig_ids = [c["id"] for c in categories]
    orig_ids_sorted = sorted(orig_ids)

    cat_id_to_idx = {}   # original -> contiguous [1..K]
    idx_to_cat_id = {}   # contiguous -> original
    cat_id_to_name = {}  # original -> name

    for c in categories:
        cat_id_to_name[c["id"]] = c["name"]

    for i, cid in enumerate(orig_ids_sorted, start=1):
        cat_id_to_idx[cid] = i
        idx_to_cat_id[i] = cid

    num_classes = len(orig_ids_sorted) + 1  # +1 for background
    return cat_id_to_idx, idx_to_cat_id, num_classes, categories, cat_id_to_name


def get_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            loss_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] "
                  f"Total loss: {losses.item():.4f} | {loss_str}")


@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=None):
    was_training = model.training
    model.train()

    total_loss = 0.0
    count = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        count += 1

        if max_batches is not None and (i + 1) >= max_batches:
            break

    if not was_training:
        model.eval()

    if count == 0:
        return None
    return total_loss / count


@torch.no_grad()
def evaluate_map(model, data_loader, device, test_ann_file, idx_to_cat_id):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    coco_gt = COCO(test_ann_file)
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                label_idx = int(label.item())
                if label_idx not in idx_to_cat_id:
                    continue
                category_id = idx_to_cat_id[label_idx]

                results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": float(score.item()),
                })

    if not results:
        print("No detections produced by the model; cannot compute mAP.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precompute tree+classical augmentations once in-memory "
                    "(CLASSICAL nodes disallowed, NONE allowed), then train Faster R-CNN "
                    "and track best mAP@0.50."
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
        help="Number of augmented versions per training image (in addition to the original).",
    )
    parser.add_argument(
        "--tree-levels",
        type=int,
        default=2,
        help="Depth of the augmentation tree.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help=(
            "Path to save fine-tuned model. If not set, defaults to "
            "<base_output_root>/<subset_path>/fsod_fewshot_fasterrcnn.pth"
        ),
    )
    args = parser.parse_args()


        # ------------------------------------------------------------
    # W&B INIT
    # ------------------------------------------------------------
    wandb.init(
        project="fsod-tree-augmentation",
        config={
            "subset_path": args.subset_path,
            "base_output_root": args.base_output_root,
            "num_augmentations": args.num_augmentations,
            "tree_levels": args.tree_levels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    )



    subset_root = os.path.join(args.base_output_root, args.subset_path)
    print("Using subset root:", subset_root)

    train_root = os.path.join(subset_root, "train")
    test_root = os.path.join(subset_root, "test")
    train_ann_file = os.path.join(train_root, "annotations.json")
    test_ann_file = os.path.join(test_root, "annotations.json")

    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f"Train annotations not found at {train_ann_file}")
    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"Test annotations not found at {test_ann_file}")

    # Category mapping
    (
        cat_id_to_idx,
        idx_to_cat_id,
        num_classes,
        categories,
        cat_id_to_name,
    ) = build_category_mapping(train_ann_file)

    print(f"Found {num_classes - 1} categories in train split.")
    print("Categories:", [c["name"] for c in categories])

    # Augmentation tree
    print(f"Creating augmentation tree with {args.tree_levels} levels "
          f"(CLASSICAL disallowed, NONE allowed)...")
    augmentation_tree = create_augmentation_tree_no_classical(args.tree_levels)
    print("Augmentation tree structure:")
    print(augmentation_tree)

    # Log tree as text to wandb
    wandb.log({"augmentation_tree": str(augmentation_tree)})


    # Classical transform (ColorJitter)
    classical_transform = get_classical_augmentations()
    color_jitter = classical_transform.transforms[0] if classical_transform.transforms else None

    # Datasets (train is fully precomputed in memory)
    train_dataset = PrecomputedAugmentedFSODTrainDataset(
        split_root=train_root,
        ann_file=train_ann_file,
        cat_id_to_idx=cat_id_to_idx,
        cat_id_to_name=cat_id_to_name,
        augmentation_tree=augmentation_tree,
        num_augmentations=args.num_augmentations,
        color_jitter=color_jitter,
    )

    test_dataset = FSODFewShotTestDataset(
        split_root=test_root,
        ann_file=test_ann_file,
        cat_id_to_idx=cat_id_to_idx,
    )

    # DataLoaders — num_workers=0 to avoid CUDA + fork issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model, optimizer, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_ap50 = -1.0
    best_epoch = -1

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} ====")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Quick test loss (optional)
        val_loss = evaluate(model, test_loader, device, max_batches=10)
        if val_loss is not None:
            print(f"[Eval-Loss] Epoch {epoch} - Approx. test loss: {val_loss:.4f}")

        # mAP evaluation
        print(f"[Eval-mAP] Epoch {epoch} - Running COCO-style mAP evaluation on test split...")
        stats = evaluate_map(
            model=model,
            data_loader=test_loader,
            device=device,
            test_ann_file=test_ann_file,
            idx_to_cat_id=idx_to_cat_id,
        )

        if stats is not None:
            ap = stats[0]
            ap50 = stats[1]
            ap75 = stats[2]
            ar = stats[6]
            print(f"[Eval-mAP] Epoch {epoch} stats (bbox):")
            print(f"  AP @[IoU=0.50:0.95]: {ap:.4f}")
            print(f"  AP @[IoU=0.50]:       {ap50:.4f}")
            print(f"  AP @[IoU=0.75]:       {ap75:.4f}")
            print(f"  AR @[IoU=0.50:0.95]:  {ar:.4f}")

            if ap50 > best_ap50:
                best_ap50 = ap50
                best_epoch = epoch
                print(f"  --> New best AP50 = {best_ap50:.4f} at epoch {best_epoch}")

            wandb.log({
                "epoch": epoch,
                "train_loss": float(losses.item()) if 'losses' in locals() else None,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "mAP": float(ap),
                "mAP50": float(ap50),
                "mAP75": float(ap75),
                "AR": float(ar),
                "best_mAP50": float(best_ap50),
            })


    # Save final model
    if args.output_model is None:
        output_model = os.path.join(subset_root, "fsod_fewshot_fasterrcnn.pth")
    else:
        output_model = args.output_model

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"\nSaved fine-tuned model to: {output_model}")

    # Best epoch
    if best_epoch == -1:
        print("No valid mAP50 scores were computed (no detections?).")
    else:
        print(f"\nBest mAP@0.50 achieved at epoch {best_epoch} with AP50 = {best_ap50:.4f}")
        wandb.log({
		"final_best_epoch": best_epoch,
		"final_best_AP50": best_ap50,
	})
        wandb.finish()



if __name__ == "__main__":
    random.seed(time.time())
    main()
