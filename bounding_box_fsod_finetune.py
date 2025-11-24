import argparse
import json
import os
from typing import Dict, Any, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------------
# Dataset
# -----------------------------
class FSODFewShotDataset(Dataset):
    def __init__(
        self,
        split_root: str,
        ann_file: str,
        cat_id_to_idx: Dict[int, int],
    ):
        """
        split_root: path to train/ or test/ directory
        ann_file: path to annotations.json for this split
        cat_id_to_idx: maps original category_id -> [1..K] contiguous indices
        """
        self.split_root = split_root
        self.img_root = os.path.join(split_root, "images")
        self.cat_id_to_idx = cat_id_to_idx

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Build image_id -> list[annotations]
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
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, w, h]
            # convert to [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            boxes.append([x1, y1, x2, y2])

            orig_cat_id = ann["category_id"]
            labels.append(self.cat_id_to_idx[orig_cat_id])

            area = ann.get("area", w * h)
            areas.append(area)

            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        image_id_tensor = torch.tensor([img_id])

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id_tensor,
            "area": areas,
            "iscrowd": iscrowd,
        }

        # Basic transform: convert to tensor
        img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Utility to build category mapping
# -----------------------------
def build_category_mapping(train_ann_file: str):
    with open(train_ann_file, "r") as f:
        coco = json.load(f)

    categories = coco["categories"]
    orig_ids = [c["id"] for c in categories]
    orig_ids_sorted = sorted(orig_ids)

    cat_id_to_idx = {}  # original -> contiguous [1..K]
    idx_to_cat_id = {}

    for i, cid in enumerate(orig_ids_sorted, start=1):
        cat_id_to_idx[cid] = i
        idx_to_cat_id[i] = cid

    num_classes = len(orig_ids_sorted) + 1  # +1 for background
    return cat_id_to_idx, idx_to_cat_id, num_classes, categories


# -----------------------------
# Model setup
# -----------------------------
def get_model(num_classes: int):
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head with a new one (background + num_classes-1)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# -----------------------------
# Training loop
# -----------------------------
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
    # Save the original training/eval state
    was_training = model.training
    # We need the model in train mode to get a loss dict from torchvision detection models
    model.train()

    total_loss = 0.0
    count = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # returns a dict in train mode
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        count += 1

        if max_batches is not None and (i + 1) >= max_batches:
            break

    # Restore original state
    if not was_training:
        model.eval()

    if count == 0:
        return None
    return total_loss / count


@torch.no_grad()
def evaluate_map(model, data_loader, device, test_ann_file, idx_to_cat_id):
    """
    Compute COCO-style mAP on the test set.

    model: trained detection model
    data_loader: DataLoader for test split
    device: cuda/cpu
    test_ann_file: path to test/annotations.json
    idx_to_cat_id: mapping from contiguous label (1..K) back to original category_id
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    coco_gt = COCO(test_ann_file)

    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)  # list of predictions

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                # Map contiguous label back to original COCO category_id
                label_idx = int(label.item())
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

    # coco_eval.stats:
    # [0] AP @[0.5:0.95], [1] AP50, [2] AP75, [3] AP small, [4] AP medium, [5] AP large,
    # [6] AR @[0.5:0.95], [7] AR small, [8] AR medium, [9] AR large
    return coco_eval.stats




# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Faster R-CNN on a few-shot FSOD subset."
    )
    parser.add_argument(
        "--subset-path",
        type=str,
        required=True,
        help="Relative path to the subset directory (e.g., 'subset_43', 'classical/subset_47').",
    )
    parser.add_argument(
        "--base-output-root",
        type=str,
        default="few_shot_datasets/bounding_boxes/fsod",
        help="Base directory containing subset folders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
    # Build subset path by joining base_output_root with subset_path
    # ------------------------------------------------------------
    subset_root = os.path.join(args.base_output_root, args.subset_path)
    print("Using subset root:", subset_root)

    train_root = os.path.join(subset_root, "train")
    test_root = os.path.join(subset_root, "test")
    train_ann_file = os.path.join(train_root, "annotations.json")
    test_ann_file = os.path.join(test_root, "annotations.json")

    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f"Train annotations not found at {train_ann_file}")

    # Build category mapping and get num_classes
    cat_id_to_idx, idx_to_cat_id, num_classes, categories = build_category_mapping(
        train_ann_file
    )
    print(f"Found {num_classes - 1} categories in train split.")
    print("Categories:", [c["name"] for c in categories])

    # Datasets
    train_dataset = FSODFewShotDataset(
        split_root=train_root,
        ann_file=train_ann_file,
        cat_id_to_idx=cat_id_to_idx,
    )
    test_dataset = FSODFewShotDataset(
        split_root=test_root,
        ann_file=test_ann_file,
        cat_id_to_idx=cat_id_to_idx,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Model, optimizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Optional quick test loss
        val_loss = evaluate(model, test_loader, device, max_batches=10)
        if val_loss is not None:
            print(f"[Eval] Epoch {epoch} - Approx. test loss: {val_loss:.4f}")


    # After training loop, compute mAP on test split
    print("Running COCO-style mAP evaluation on test split...")
    stats = evaluate_map(
        model=model,
        data_loader=test_loader,
        device=device,
        test_ann_file=test_ann_file,
        idx_to_cat_id=idx_to_cat_id,
    )

    if stats is not None:
        print("COCO evaluation stats (bbox):")
        print(f"AP @[IoU=0.50:0.95]: {stats[0]:.4f}")
        print(f"AP @[IoU=0.50]:       {stats[1]:.4f}")
        print(f"AP @[IoU=0.75]:       {stats[2]:.4f}")
        print(f"AR @[IoU=0.50:0.95]: {stats[6]:.4f}")


    # Save model
    if args.output_model is None:
        output_model = os.path.join(subset_root, "fsod_fewshot_fasterrcnn.pth")
    else:
        output_model = args.output_model

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"Saved fine-tuned model to: {output_model}")


if __name__ == "__main__":
    main()