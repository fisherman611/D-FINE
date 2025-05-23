import os
import json 

images_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/train/images'
labels_dir = 'Fisheye8K/Fisheye8K_all_including_train&test/train/labels'
output_json = 'Fisheye8K/Fisheye8K_all_including_train&test/train/train.json'

# List your class names in order (e.g., ["person", "car", ...])
class_names = ["Bus", "Bike", "Car","Pedestrian","Truck"]  # <-- CHANGE THIS

# Prepare COCO structure
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories
for i, name in enumerate(class_names):
    coco["categories"].append({"id": i, "name": name})

annotation_id = 1
image_id = 1

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(images_dir, img_file)
    # You may want to use PIL to get image size
    from PIL import Image        #type: ignore
    with Image.open(img_path) as img:
        width, height = img.size

    coco["images"].append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
    if os.path.exists(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                # Convert normalized to absolute
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x = x_center - w / 2
                y = y_center - h / 2
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
    image_id += 1

with open(output_json, "w") as f:
    json.dump(coco, f)