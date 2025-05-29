import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def load_coco_annotations(json_path):
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_annotations(data, json_path):
    """Save COCO format annotations"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def sample_and_split_dataset(
    source_annotations_path,
    source_images_dir,
    output_base_dir,
    sample_ratio=1,
    train_ratio=0.8,
    random_seed=42.
):
    """
    Sample 100% of the dataset and split into train/val

    Args:
        source_annotations_path: Path to instances_train.json
        source_images_dir: Path to train/images directory
        output_base_dir: Base directory for output
        sample_ratio: Ratio of data to sample (default 0.05 for 5%)
        train_ratio: Ratio for train split (default 0.8 for 80%)
        random_seed: Random seed for reproducibility
    """

    random.seed(random_seed)

    # Load original annotations
    print("Loading annotations...")
    coco_data = load_coco_annotations(source_annotations_path)

    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]

    # Sample 5% of images
    sample_size = max(1, int(len(image_ids) * sample_ratio))
    sampled_image_ids = random.sample(image_ids, sample_size)

    print(f"Original dataset: {len(image_ids)} images")
    print(f"Sampled dataset: {len(sampled_image_ids)} images ({sample_ratio*100}%)")

    # Split sampled images into train/val
    train_size = int(len(sampled_image_ids) * train_ratio)
    train_image_ids = set(sampled_image_ids[:train_size])
    val_image_ids = set(sampled_image_ids[train_size:])

    print(f"Train split: {len(train_image_ids)} images ({train_ratio*100}%)")
    print(f"Val split: {len(val_image_ids)} images ({(1-train_ratio)*100}%)")

    # Create image ID to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Filter images and annotations for each split
    for split_name, split_image_ids in [('train', train_image_ids), ('val', val_image_ids)]:
        print(f"\nProcessing {split_name} split...")

        # Filter images
        split_images = [img for img in coco_data['images'] if img['id'] in split_image_ids]

        # Filter annotations
        split_annotations = [ann for ann in coco_data['annotations']
                           if ann['image_id'] in split_image_ids]

        # Create new COCO data structure
        split_data = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco_data['categories'],
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', [])
        }

        # Create output directories
        split_dir = Path(output_base_dir) / split_name
        split_images_dir = split_dir / 'images'
        split_annotations_dir = split_dir / 'annotations'
        split_labels_dir = split_dir / 'labels'

        for dir_path in [split_images_dir, split_annotations_dir, split_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save annotations JSON
        annotations_json_path = split_dir / f'{split_name}.json'
        save_coco_annotations(split_data, annotations_json_path)

        # Copy images
        print(f"Copying {len(split_image_ids)} images...")
        for image_id in split_image_ids:
            filename = id_to_filename[image_id]
            source_path = Path(source_images_dir) / filename
            dest_path = split_images_dir / filename

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
            else:
                print(f"Warning: Image not found: {source_path}")

        # Create empty annotations and labels directories (maintaining structure)
        # You can add logic here to copy/process annotation files if they exist
        print(f"Created directory structure for {split_name}")

    print(f"\nDataset processing complete!")
    print(f"Output saved to: {output_base_dir}")

def main():
    # Configuration
    SOURCE_ANNOTATIONS = "Fisheye8K/Fisheye8K_all_including_train&test/annotation_1/instances_train.json"
    SOURCE_IMAGES = "Fisheye8K/Fisheye8K_all_including_train&test/train/images"
    OUTPUT_BASE_DIR = "Fisheye8K/Fisheye8K_all_including_train&test/full_dataset"

    # Verify source files exist
    if not os.path.exists(SOURCE_ANNOTATIONS):
        print(f"Error: Annotations file not found: {SOURCE_ANNOTATIONS}")
        return

    if not os.path.exists(SOURCE_IMAGES):
        print(f"Error: Images directory not found: {SOURCE_IMAGES}")
        return

    # Run the sampling and splitting
    sample_and_split_dataset(
        source_annotations_path=SOURCE_ANNOTATIONS,
        source_images_dir=SOURCE_IMAGES,
        output_base_dir=OUTPUT_BASE_DIR,
        sample_ratio=1,  # Full data%
        train_ratio=0.8,    # 80% train, 20% val
        random_seed=42
    )

    print("\nFinal directory structure:")
    print("sampled_dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   ├── annotations/")
    print("│   ├── labels/")
    print("│   └── train.json")
    print("└── val/")
    print("    ├── images/")
    print("    ├── annotations/")
    print("    ├── labels/")
    print("    └── val.json")

if __name__ == "__main__":
    main()