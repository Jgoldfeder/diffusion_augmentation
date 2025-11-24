import json
import os
import random
import argparse
import shutil
from typing import Dict, List, Set, Any


def build_index(ann_data: Dict[str, Any]):
    images = ann_data["images"]
    annotations = ann_data["annotations"]
    categories = ann_data["categories"]

    image_by_id: Dict[int, Dict[str, Any]] = {img["id"]: img for img in images}
    category_by_id: Dict[int, Dict[str, Any]] = {cat["id"]: cat for cat in categories}

    # category_id -> set of image_ids
    images_by_category: Dict[int, Set[int]] = {}
    for ann in annotations:
        cid = ann["category_id"]
        iid = ann["image_id"]
        images_by_category.setdefault(cid, set()).add(iid)

    return image_by_id, category_by_id, images_by_category


def choose_categories(images_by_category: Dict[int, Set[int]],
                      num_categories: int,
                      rng: random.Random) -> List[int]:
    # categories must have at least 2 images to support 2-shot train
    eligible = [cid for cid, img_ids in images_by_category.items() if len(img_ids) >= 2]
    if len(eligible) < num_categories:
        raise ValueError(
            f"Not enough categories with at least 2 images. "
            f"Needed {num_categories}, found {len(eligible)}."
        )
    return rng.sample(eligible, num_categories)


def split_images_for_categories(
    selected_cats: List[int],
    images_by_category: Dict[int, Set[int]],
    rng: random.Random
):
    train_image_ids: Set[int] = set()
    test_image_ids: Set[int] = set()

    for cid in selected_cats:
        img_ids = list(images_by_category[cid])
        if len(img_ids) < 2:
            # Should not happen due to filtering, but guard anyway
            continue

        rng.shuffle(img_ids)
        train_for_cat = img_ids[:2]
        test_for_cat = img_ids[2:]

        train_image_ids.update(train_for_cat)
        test_image_ids.update(test_for_cat)

    # Ensure no image ends up in both train and test:
    # if an image is used for train in any class, it is train-only.
    test_image_ids.difference_update(train_image_ids)

    return train_image_ids, test_image_ids


def filter_annotations(
    ann_data: Dict[str, Any],
    selected_cats: List[int],
    train_image_ids: Set[int],
    test_image_ids: Set[int]
):
    annotations = ann_data["annotations"]
    images = ann_data["images"]
    categories = ann_data["categories"]

    selected_cats_set = set(selected_cats)

    # Filter categories down to the selected ones
    filtered_categories = [c for c in categories if c["id"] in selected_cats_set]

    # Filter images for each split
    train_images = [img for img in images if img["id"] in train_image_ids]
    test_images = [img for img in images if img["id"] in test_image_ids]

    # Filter annotations by both image_id and category_id
    train_annotations = [
        ann for ann in annotations
        if ann["image_id"] in train_image_ids and ann["category_id"] in selected_cats_set
    ]
    test_annotations = [
        ann for ann in annotations
        if ann["image_id"] in test_image_ids and ann["category_id"] in selected_cats_set
    ]

    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": filtered_categories,
    }
    test_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": filtered_categories,
    }

    return train_data, test_data


def copy_images(
    subset_root: str,
    split_name: str,
    images: List[Dict[str, Any]],
    dataset_root: str,
):
    img_root = os.path.join(subset_root, split_name, "images")
    os.makedirs(img_root, exist_ok=True)

    for img in images:
        # file_name already something like "part_1/xxx.jpg" or "part_2/yyy.jpg"
        rel_path = img["file_name"]
        src = os.path.join(dataset_root, rel_path)
        dst = os.path.join(img_root, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.exists(src):
            print(f"[WARNING] Missing source image: {src}")
            continue

        shutil.copy2(src, dst)


def save_annotations(subset_root: str, split_name: str, ann_data: Dict[str, Any]):
    split_root = os.path.join(subset_root, split_name)
    os.makedirs(split_root, exist_ok=True)
    out_path = os.path.join(split_root, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(ann_data, f)
    print(f"Saved {split_name} annotations to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a few-shot FSOD subset with 5 random classes."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (used to name the subset directory).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./torch/fsod",
        help="Root of the FSOD dataset (containing part_1, part_2, annotations, etc.).",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="annotations/fsod_train.json",
        help="Path to the FSOD train annotation JSON, relative to dataset-root or absolute.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="few_shot_datasets/bounding_boxes/fsod",
        help="Root directory where few-shot subsets will be saved.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of random classes to sample (default: 5).",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Resolve paths
    if os.path.isabs(args.train_json):
        train_json_path = args.train_json
    else:
        train_json_path = os.path.join(args.dataset_root, args.train_json)

    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Could not find train JSON at {train_json_path}")

    with open(train_json_path, "r") as f:
        ann_data = json.load(f)

    image_by_id, category_by_id, images_by_category = build_index(ann_data)

    # Sample categories
    selected_cats = choose_categories(images_by_category, args.num_classes, rng)
    print("Selected category IDs:", selected_cats)
    print("Selected category names:", [category_by_id[cid]["name"] for cid in selected_cats])

    # Split images into train/test for these categories
    train_image_ids, test_image_ids = split_images_for_categories(
        selected_cats, images_by_category, rng
    )
    print(f"Number of train images: {len(train_image_ids)}")
    print(f"Number of test images: {len(test_image_ids)}")

    # Build filtered COCO-style annotations for train and test
    train_data, test_data = filter_annotations(
        ann_data=ann_data,
        selected_cats=selected_cats,
        train_image_ids=train_image_ids,
        test_image_ids=test_image_ids,
    )

    subset_root = os.path.join(args.output_root, f"subset_{args.seed}")
    os.makedirs(subset_root, exist_ok=True)

    # Copy images
    copy_images(subset_root, "train", train_data["images"], args.dataset_root)
    copy_images(subset_root, "test", test_data["images"], args.dataset_root)

    # Save annotations
    save_annotations(subset_root, "train", train_data)
    save_annotations(subset_root, "test", test_data)

    # Save subset metadata
    meta = {
        "seed": args.seed,
        "selected_category_ids": selected_cats,
        "selected_category_names": [category_by_id[cid]["name"] for cid in selected_cats],
        "num_train_images": len(train_data["images"]),
        "num_test_images": len(test_data["images"]),
    }
    meta_path = os.path.join(subset_root, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved subset metadata to {meta_path}")


if __name__ == "__main__":
    main()
