#!/usr/bin/env python3
"""
Korean VQA Data Preprocessing Script

Prepares the Korean VLM data for Qwen2-VL fine-tuning:
1. Loads all JSONL files from the source directory
2. Creates train/val/test splits
3. Optionally samples a subset for faster experimentation
4. Saves in Qwen2-VL compatible format
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import shutil

# Data source configuration
DATA_ROOT = "/NetDisk/ingyu/VLM_DATA/vlm_20251118"
LABEL_DIR = os.path.join(DATA_ROOT, "label_v2")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

# Dataset categories for balanced sampling
DATASET_CATEGORIES = {
    "korean_problem": [
        "AIHUB_koreanproblem_v2_axolotl.jsonl",
    ],
    "math_problem": [
        "AIHUB_mathproblem_multiple_v2_axolotl.jsonl",
        "AIHUB_mathproblem_subjective_v2_axolotl.jsonl",
    ],
    "visualization": [
        "AIHUB_Visualization_v2_axolotl.jsonl",
        "bichallava_instruct_230k_chart_v2_axolotl.jsonl",
    ],
    "table_vqa": [
        "table-VQA-ko-60k.jsonl",
    ],
    "captioning": [
        "llava-ko-recap-120k_v2_axolotl.jsonl",
        "out-kor-llava_v2_axolotl.jsonl",
    ],
    "document": [
        "AIHUB_subjectmaterial_image_modify_v2_axolotl.jsonl",
        "AIHUB_subjectmaterial_text_modify_v2_axolotl.jsonl",
    ],
    "arxiv": [
        "arxiv_kor_v2_axolotl.jsonl",
        # "arxiv_eng_v2_axolotl.jsonl",  # Skip English for Korean-focused training
    ],
    "latex": [
        "LaTeX-update_v2_axolotl.jsonl",
    ],
    "general": [
        "image_folder_1_v2_axolotl.jsonl",
        "image_folder_2_v2_axolotl.jsonl",
        "kor1_YiSang_hq.jsonl",
    ],
}


def load_jsonl(filepath: str, max_samples: int = None) -> List[Dict]:
    """Load JSONL file with optional sample limit."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def validate_image_path(item: Dict, image_root: str) -> bool:
    """Check if the image file exists."""
    try:
        messages = item.get("messages", [])
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "image":
                        img_path = c.get("path", "")
                        full_path = os.path.join(image_root, img_path)
                        if not os.path.exists(full_path):
                            return False
        return True
    except Exception:
        return False


def update_image_paths(item: Dict, image_root: str) -> Dict:
    """Update image paths to be absolute or relative to new root."""
    item = json.loads(json.dumps(item))  # Deep copy

    messages = item.get("messages", [])
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "image":
                    # Keep relative path but ensure it starts with images/
                    img_path = c.get("path", "")
                    if not img_path.startswith("images/"):
                        c["path"] = os.path.join("images", img_path)

    return item


def split_data(
    data: List[Dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets."""
    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)

    n = len(data_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data_shuffled[:train_end]
    val_data = data_shuffled[train_end:val_end]
    test_data = data_shuffled[val_end:]

    return train_data, val_data, test_data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} samples to {filepath}")


def prepare_data(
    output_dir: str,
    max_samples_per_category: int = None,
    total_max_samples: int = None,
    validate_images: bool = False,
    seed: int = 42
):
    """
    Main data preparation function.

    Args:
        output_dir: Directory to save processed data
        max_samples_per_category: Max samples per category (for balanced sampling)
        total_max_samples: Total max samples across all categories
        validate_images: Whether to validate image file existence
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("=" * 70)
    print("Korean VQA Data Preparation")
    print("=" * 70)
    print(f"Source: {DATA_ROOT}")
    print(f"Output: {output_dir}")
    print(f"Max per category: {max_samples_per_category or 'unlimited'}")
    print(f"Total max: {total_max_samples or 'unlimited'}")
    print()

    all_data = []
    category_counts = defaultdict(int)

    # Load data from each category
    for category, files in DATASET_CATEGORIES.items():
        print(f"\n[{category}]")
        category_data = []

        for filename in files:
            filepath = os.path.join(LABEL_DIR, filename)
            if not os.path.exists(filepath):
                print(f"  Warning: {filename} not found, skipping")
                continue

            print(f"  Loading: {filename}...", end=" ")
            data = load_jsonl(filepath)
            print(f"{len(data)} samples")

            # Validate and update image paths
            valid_data = []
            for item in data:
                item = update_image_paths(item, DATA_ROOT)
                if validate_images:
                    if validate_image_path(item, DATA_ROOT):
                        valid_data.append(item)
                else:
                    valid_data.append(item)

            if validate_images and len(valid_data) < len(data):
                print(f"    (Filtered: {len(data)} -> {len(valid_data)} after image validation)")

            category_data.extend(valid_data)

        # Sample if max_samples_per_category is set
        if max_samples_per_category and len(category_data) > max_samples_per_category:
            random.shuffle(category_data)
            category_data = category_data[:max_samples_per_category]
            print(f"  Sampled: {max_samples_per_category} samples")

        category_counts[category] = len(category_data)
        all_data.extend(category_data)

    # Apply total max samples if set
    if total_max_samples and len(all_data) > total_max_samples:
        random.shuffle(all_data)
        all_data = all_data[:total_max_samples]
        print(f"\nTotal sampled: {total_max_samples} samples")

    print(f"\n{'=' * 70}")
    print(f"Total samples: {len(all_data)}")
    print(f"{'=' * 70}")

    # Print category distribution
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Split into train/val/test
    print("\nSplitting data...")
    train_data, val_data, test_data = split_data(all_data, seed=seed)

    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    # Save splits
    print("\nSaving splits...")
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))

    # Save metadata
    metadata = {
        "source": DATA_ROOT,
        "total_samples": len(all_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "category_counts": dict(category_counts),
        "max_samples_per_category": max_samples_per_category,
        "total_max_samples": total_max_samples,
        "seed": seed,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to {metadata_path}")

    # Create symlink to images directory
    images_link = os.path.join(output_dir, "images")
    if not os.path.exists(images_link):
        os.symlink(IMAGE_DIR, images_link)
        print(f"Created symlink: {images_link} -> {IMAGE_DIR}")

    print(f"\n{'=' * 70}")
    print("Data preparation complete!")
    print(f"{'=' * 70}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare Korean VQA data for fine-tuning")

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/data/korean_vqa",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Maximum samples per category (for balanced sampling)"
    )
    parser.add_argument(
        "--total-max",
        type=int,
        default=None,
        help="Total maximum samples"
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate that image files exist"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10K samples per category, 100K total"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Tiny mode: 1K samples per category, 10K total (for testing)"
    )

    args = parser.parse_args()

    # Apply preset modes
    max_per_cat = args.max_per_category
    total_max = args.total_max

    if args.tiny:
        max_per_cat = 1000
        total_max = 10000
        args.output = args.output.rstrip('/') + "_tiny"
    elif args.quick:
        max_per_cat = 10000
        total_max = 100000
        args.output = args.output.rstrip('/') + "_quick"

    prepare_data(
        output_dir=args.output,
        max_samples_per_category=max_per_cat,
        total_max_samples=total_max,
        validate_images=args.validate_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
