import os 
import random
from shutil import copy2
import json 

# === CONFIGURATION ===
# Change these paths as needed
original_images_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/train/images'
original_json = 'Fisheye8K/Fisheye8K_all_including_train&test/train/train.json'
train_images_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/train/images'  # (train images stay here)
val_images_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/val/images'
train_json_out = 'Fisheye8K/Fisheye8K_all_including_train&test/train/train.json'
val_json_out = 'Fisheye8K/Fisheye8K_all_including_train&test/val/val.json'

val_ratio = 0.2  # 20% for validation

# === LOAD COCO ANNOTATION ===
with open(original_json, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# === SPLIT IMAGES ===
random.shuffle(images)
split_idx = int((1 - val_ratio) * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

train_image_ids = {img['id'] for img in train_images}
val_image_ids = {img['id'] for img in val_images}

# === SPLIT ANNOTATIONS ===
train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

# === SAVE NEW ANNOTATION FILES ===
with open(train_json_out, 'w') as f:
    json.dump({'images': train_images, 'annotations': train_annotations, 'categories': categories}, f)
os.makedirs('Fisheye8K/Fisheye8K_all_including_train&test/val', exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(os.path.dirname(val_json_out), exist_ok=True)
val_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/val'
os.makedirs(val_dir, exist_ok=True)
val_json_out = os.path.join(val_dir, 'val.json')
with open(val_json_out, 'w') as f:
    json.dump({'images': val_images, 'annotations': val_annotations, 'categories': categories}, f)

# === MOVE/COPY VALIDATION IMAGES ===
for img in val_images:
    src = os.path.join(original_images_dir, img['file_name'])
    dst = os.path.join(val_images_dir, img['file_name'])
    if os.path.exists(src):
        copy2(src, dst)  # Use copy2 to keep original train images, or use shutil.move to move instead

print(f"Split complete! {len(train_images)} train images, {len(val_images)} val images.")
print(f"New annotation files: {train_json_out}, {val_json_out}")