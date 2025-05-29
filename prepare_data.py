#!/usr/bin/env python3
"""
Dataset to COCO Format Converter

This script converts a dataset with separate JSON metadata and text-based bounding box labels
into the COCO format. The input dataset has JSON files containing image metadata and separate
.txt files with YOLO-style normalized bounding box annotations.

Input Structure:
    parent_directory/
    ├── test/
    │   ├── images/
    │   ├── labels/
    │   └── test.json
    └── train/
        ├── images/
        ├── labels/
        └── train.json

Output: Complete COCO-formatted JSON files (instances_train.json, instances_test.json)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os 

def load_json_metadata(json_path: str) -> Dict:
    """
    Load the base JSON metadata file containing COCO-like structure.

    Args:
        json_path: Path to the JSON file (train.json or test.json)

    Returns:
        Dictionary containing the loaded JSON data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded metadata from {json_path}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: JSON file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON format in {json_path}: {e}")
        raise


def create_category_lookup(categories: List[Dict]) -> Dict[int, str]:
    """
    Create a lookup dictionary mapping category IDs to category names.

    Args:
        categories: List of category dictionaries from COCO JSON

    Returns:
        Dictionary mapping category_id -> category_name
    """
    category_lookup = {}
    for category in categories:
        category_lookup[category['id']] = category['name']

    print(f"✓ Created category lookup with {len(category_lookup)} categories:")
    for cat_id, cat_name in category_lookup.items():
        print(f"   - ID {cat_id}: {cat_name}")

    return category_lookup


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Parse a single line from a YOLO-style label file.

    Args:
        line: Single line from .txt file in format:
              [line_index] [class_id] [x_center_norm] [y_center_norm] [width_norm] [height_norm]

    Returns:
        Tuple of (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
        or None if parsing fails
    """
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"   ✗ Warning: Expected 6 values, got {len(parts)}: {line.strip()}")
            return None

        # Parse components (ignoring line_index at position 0)
        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])

        # Validate normalized coordinates are in [0, 1] range
        coords = [x_center_norm, y_center_norm, width_norm, height_norm]
        if not all(0.0 <= coord <= 1.0 for coord in coords):
            print(f"   ✗ Warning: Normalized coordinates out of range [0,1]: {coords}")
            return None

        return class_id, x_center_norm, y_center_norm, width_norm, height_norm

    except (ValueError, IndexError) as e:
        print(f"   ✗ Warning: Failed to parse line '{line.strip()}': {e}")
        return None


def yolo_to_coco_bbox(x_center_norm: float, y_center_norm: float,
                      width_norm: float, height_norm: float,
                      img_width: int, img_height: int) -> List[int]:
    """
    Convert YOLO-style normalized bounding box to COCO format.

    YOLO format: [x_center_normalized, y_center_normalized, width_normalized, height_normalized]
    COCO format: [x_top_left, y_top_left, width, height] in pixels

    Args:
        x_center_norm: Normalized x-coordinate of bounding box center (0-1)
        y_center_norm: Normalized y-coordinate of bounding box center (0-1)
        width_norm: Normalized width of bounding box (0-1)
        height_norm: Normalized height of bounding box (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of [x, y, width, height] in COCO format (pixels, integers)
    """
    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center_norm * img_width
    y_center_px = y_center_norm * img_height
    width_px = width_norm * img_width
    height_px = height_norm * img_height

    # Convert center coordinates to top-left corner coordinates
    x_top_left = x_center_px - (width_px / 2)
    y_top_left = y_center_px - (height_px / 2)

    # Round to integers and ensure non-negative values
    x_top_left = max(0, round(x_top_left))
    y_top_left = max(0, round(y_top_left))
    width_px = max(1, round(width_px))  # Ensure minimum width of 1
    height_px = max(1, round(height_px))  # Ensure minimum height of 1

    return [x_top_left, y_top_left, width_px, height_px]


def calculate_bbox_area(bbox: List[int]) -> int:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box in COCO format [x, y, width, height]

    Returns:
        Area of the bounding box in pixels
    """
    return bbox[2] * bbox[3]  # width * height


def process_label_file(label_path: str, image_info: Dict,
                      category_lookup: Dict[int, str]) -> List[Dict]:
    """
    Process a single label file and convert annotations to COCO format.

    Args:
        label_path: Path to the .txt label file
        image_info: Dictionary containing image metadata (id, width, height, etc.)
        category_lookup: Dictionary mapping category IDs to names

    Returns:
        List of annotation dictionaries in COCO format
    """
    annotations = []

    if not os.path.exists(label_path):
        print(f"   ⚠ Label file not found: {label_path}")
        return annotations

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"   ✓ Processing {len(lines)} annotations from {os.path.basename(label_path)}")

        for line_num, line in enumerate(lines, 1):
            if not line.strip():  # Skip empty lines
                continue

            parsed = parse_yolo_line(line)
            if parsed is None:
                print(f"   ✗ Skipping invalid line {line_num}")
                continue

            class_id, x_center_norm, y_center_norm, width_norm, height_norm = parsed

            # Validate category ID exists
            if class_id not in category_lookup:
                print(f"   ✗ Warning: Unknown category ID {class_id} on line {line_num}")
                continue

            # Convert YOLO bbox to COCO format
            coco_bbox = yolo_to_coco_bbox(
                x_center_norm, y_center_norm, width_norm, height_norm,
                image_info['width'], image_info['height']
            )

            # Calculate area
            area = calculate_bbox_area(coco_bbox)

            # Create annotation dictionary
            annotation = {
                'id': len(annotations) + 1,  # Will be updated with global ID later
                'image_id': image_info['id'],
                'category_id': class_id,
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0,
                'segmentation': []  # Empty for bounding box annotations
            }

            annotations.append(annotation)

    except Exception as e:
        print(f"   ✗ Error processing label file {label_path}: {e}")

    return annotations


def convert_split_to_coco(split_dir: str, split_name: str,
                         output_dir: str) -> bool:
    """
    Convert a single dataset split (train/test) to COCO format.

    Args:
        split_dir: Path to the split directory (contains images/, labels/, and .json file)
        split_name: Name of the split ('train' or 'test')
        output_dir: Directory to save the output COCO JSON file

    Returns:
        True if conversion was successful, False otherwise
    """
    print(f"\n🔄 Processing {split_name} split...")

    # Define paths
    json_path = os.path.join(split_dir, f"{split_name}.json")
    labels_dir = os.path.join(split_dir, "labels")
    output_path = os.path.join(output_dir, f"instances_{split_name}.json")

    try:
        # Load base JSON metadata
        coco_data = load_json_metadata(json_path)

        # Validate required sections exist
        required_sections = ['info', 'licenses', 'categories', 'images']
        for section in required_sections:
            if section not in coco_data:
                print(f"✗ Missing required section '{section}' in {json_path}")
                return False

        # Create category lookup for validation
        category_lookup = create_category_lookup(coco_data['categories'])

        # Initialize annotations list
        coco_data['annotations'] = []
        annotation_id = 1

        # Process each image
        print(f"🖼️  Processing {len(coco_data['images'])} images...")

        for image_info in coco_data['images']:
            # Extract image filename without extension for label file matching
            file_name = image_info['file_name']
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            print(f"   📄 Processing image: {file_name}")

            # Process corresponding label file
            image_annotations = process_label_file(label_path, image_info, category_lookup)

            # Update annotation IDs and add to main list
            for annotation in image_annotations:
                annotation['id'] = annotation_id
                annotation_id += 1
                coco_data['annotations'].append(annotation)

        # Save COCO JSON file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully saved {split_name} COCO format to: {output_path}")
        print(f"   📊 Statistics:")
        print(f"      - Images: {len(coco_data['images'])}")
        print(f"      - Annotations: {len(coco_data['annotations'])}")
        print(f"      - Categories: {len(coco_data['categories'])}")

        return True

    except Exception as e:
        print(f"✗ Error processing {split_name} split: {e}")
        return False


def main():
    """
    Main function to convert dataset to COCO format.
    """
    print("🚀 Dataset to COCO Format Converter")
    print("=" * 50)

    # Configuration - MODIFY THIS PATH TO YOUR DATASET
    parent_data_directory = "Fisheye8K/Fisheye8K_all_including_train&test"  # ⚠️ UPDATE THIS PATH
    output_directory = "Fisheye8K/Fisheye8K_all_including_train&test/annotation_1"         # ⚠️ UPDATE THIS PATH

    # Validate parent directory exists
    if not os.path.exists(parent_data_directory):
        print(f"✗ Error: Parent data directory not found: {parent_data_directory}")
        print("Please update the 'parent_data_directory' variable in the script.")
        sys.exit(1)

    print(f"📁 Input directory: {parent_data_directory}")
    print(f"📁 Output directory: {output_directory}")

    # Process each split
    splits = ['train', 'test']
    success_count = 0

    for split in splits:
        split_dir = os.path.join(parent_data_directory, split)

        if not os.path.exists(split_dir):
            print(f"⚠️  Warning: {split} directory not found, skipping: {split_dir}")
            continue

        if convert_split_to_coco(split_dir, split, output_directory):
            success_count += 1

    # Final summary
    print("\n" + "=" * 50)
    print(f"🎯 Conversion completed!")
    print(f"   ✅ Successfully processed: {success_count}/{len(splits)} splits")

    if success_count > 0:
        print(f"   📂 Output files saved to: {output_directory}")
        print("   📋 Generated files:")
        for split in splits:
            output_file = os.path.join(output_directory, f"instances_{split}.json")
            if os.path.exists(output_file):
                print(f"      - instances_{split}.json")

    if success_count == 0:
        print("   ⚠️  No splits were successfully processed. Please check your input paths and file structure.")
        sys.exit(1)


if __name__ == "__main__":
    main()