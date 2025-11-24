import argparse
import json
import os
import random

from typing import Dict, Any, List

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------------------------------------------
# Category mapping (same logic as in the finetune script)
# -----------------------------------------------------------
def build_category_mapping(train_ann_file: str):
    with open(train_ann_file, "r") as f:
        coco = json.load(f)

    categories = coco["categories"]
    orig_ids = [c["id"] for c in categories]
    orig_ids_sorted = sorted(orig_ids)

    cat_id_to_idx = {}   # original -> contiguous [1..K]
    idx_to_cat_id = {}
    idx_to_cat_name = {}

    # Build mapping and name lookup
    cat_by_id = {c["id"]: c for c in categories}

    for i, cid in enumerate(orig_ids_sorted, start=1):
        cat_id_to_idx[cid] = i
        idx_to_cat_id[i] = cid
        idx_to_cat_name[i] = cat_by_id[cid]["name"]

    num_classes = len(orig_ids_sorted) + 1  # +1 for background
    return cat_id_to_idx, idx_to_cat_id, idx_to_cat_name, num_classes


# -----------------------------------------------------------
# Model setup (must match training)
# -----------------------------------------------------------
def get_model(num_classes: int, model_path: str, device: torch.device):
    # Same backbone & head structure as in training
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------
# Visualization helper
# -----------------------------------------------------------
def visualize_and_save(
    image_path: str,
    output_path: str,
    prediction: Dict[str, Any],
    idx_to_cat_name: Dict[int, str],
    score_thresh: float = 0.5,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    ax = plt.gca()

    boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()
    labels = prediction["labels"].cpu()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        label_idx = int(label.item())
        cls_name = idx_to_cat_name.get(label_idx, f"id_{label_idx}")
        text = f"{cls_name}: {score:.2f}"

        ax.text(
            x1,
            y1,
            text,
            fontsize=8,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")


# -----------------------------------------------------------
# Inference over a split
# -----------------------------------------------------------
def run_inference_on_split(
    model,
    device,
    split_root: str,
    split_name: str,
    images: List[Dict[str, Any]],
    idx_to_cat_name: Dict[int, str],
    output_base: str,
    max_images: int = None,
    score_thresh: float = 0.5,
):
    """
    model: detection model
    split_root: path to train/ or test/ directory
    split_name: 'train' or 'test'
    images: list of image dicts from annotations.json
    output_base: e.g. output_fsod/subset_i
    max_images: limit number of images (None = all)
    """
    img_root = os.path.join(split_root, "images")

    if max_images is not None:
        images = images[:max_images]

    for img_info in images:
        file_name = img_info["file_name"]
        img_path = os.path.join(img_root, file_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)

        with torch.no_grad():
            output = model([img_tensor])[0]

        vis_output_path = os.path.join(
            output_base,
            split_name,
            file_name,
        )
        visualize_and_save(
            image_path=img_path,
            output_path=vis_output_path,
            prediction=output,
            idx_to_cat_name=idx_to_cat_name,
            score_thresh=score_thresh,
        )


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize predictions of a fine-tuned FSOD few-shot model."
    )
    parser.add_argument(
        "--subset-path",
        type=str,
        required=True,
        help="Relative path to the subset directory (e.g., 'subset_0', 'classical/subset_43').",
    )
    parser.add_argument(
        "--base-output-root",
        type=str,
        default="few_shot_datasets/bounding_boxes/fsod",
        help="Base directory containing subset folders.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to the fine-tuned model .pth file. "
            "If not set, defaults to "
            "<base_output_root>/<subset_path>/fsod_fewshot_fasterrcnn.pth"
        ),
    )
    parser.add_argument(
        "--vis-output-base",
        type=str,
        default="output_fsod",
        help="Base directory to save visualizations: output_fsod/<subset_path>/...",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for visualizing predictions.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="Number of random test images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for selecting test images.",
    )

    args = parser.parse_args()

    # Build subset path by joining base_output_root with subset_path
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

    # Category mapping & number of classes
    cat_id_to_idx, idx_to_cat_id, idx_to_cat_name, num_classes = build_category_mapping(
        train_ann_file
    )
    print(f"Found {num_classes - 1} categories in train split.")
    print("Categories:", list(idx_to_cat_name.values()))

    # Model path
    if args.model_path is None:
        model_path = os.path.join(subset_root, "fsod_fewshot_fasterrcnn.pth")
    else:
        model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_model(num_classes=num_classes, model_path=model_path, device=device)

    # Load annotations (for image lists)
    with open(train_ann_file, "r") as f:
        train_coco = json.load(f)
    with open(test_ann_file, "r") as f:
        test_coco = json.load(f)

    train_images = train_coco["images"]
    test_images = test_coco["images"]

    # Prepare output directory: output_fsod/<subset_path>
    vis_output_root = os.path.join(args.vis_output_base, args.subset_path)
    os.makedirs(vis_output_root, exist_ok=True)

    # -------------------------------------------------------
    # 1) Visualize predictions on ALL train images
    # -------------------------------------------------------
    print(f"Visualizing predictions on {len(train_images)} train images...")
    run_inference_on_split(
        model=model,
        device=device,
        split_root=train_root,
        split_name="train",
        images=train_images,
        idx_to_cat_name=idx_to_cat_name,
        output_base=vis_output_root,
        max_images=None,  # all
        score_thresh=args.score_thresh,
    )

    # -------------------------------------------------------
    # 2) Visualize predictions on 5 random test images
    # -------------------------------------------------------
    rng = random.Random(args.seed)
    if len(test_images) <= args.test_samples:
        chosen_test_images = test_images
    else:
        chosen_test_images = rng.sample(test_images, args.test_samples)

    print(f"Visualizing predictions on {len(chosen_test_images)} test images...")
    run_inference_on_split(
        model=model,
        device=device,
        split_root=test_root,
        split_name="test",
        images=chosen_test_images,
        idx_to_cat_name=idx_to_cat_name,
        output_base=vis_output_root,
        max_images=None,  # we already sampled
        score_thresh=args.score_thresh,
    )

    print(f"Done. Visualizations saved under: {vis_output_root}")


if __name__ == "__main__":
    main()