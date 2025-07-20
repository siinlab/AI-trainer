"""
This module provides functionality to convert datasets between YOLOv5 and COCO formats.

Functions:
- convert_yolo_to_coco: Converts a YOLOv5 dataset to COCO format.
- convert_coco_to_yolo: Converts a COCO dataset to YOLOv5 format.
"""

import json
from pathlib import Path
import yaml
from tqdm import tqdm
from lgg import logger
import shutil


def convert_yolo_to_coco(dataset_path, output_json):
    """
    Converts a YOLOv5 dataset to COCO format.

    Args:
        dataset_path (str): Path to the YOLOv5 dataset directory containing 'images' and 'labels'.
        output_json (str): Path to the output COCO JSON file.

    Raises:
        FileNotFoundError: If the dataset does not contain 'images', 'labels', or 'data.yaml'.
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    data_yaml = dataset_path / "data.yaml"

    if not images_dir.exists() or not labels_dir.exists() or not data_yaml.exists():
        raise FileNotFoundError(
            "Dataset must contain 'images', 'labels', and 'data.yaml'."
        )

    # Load class names from data.yaml
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
        class_names = data.get("names", [])

    output_dir = Path(output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to the new dataset folder
    for image_file in images_dir.glob("*.*"):
        shutil.copy(image_file, output_images_dir / image_file.name)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(class_names)
        ],
    }

    annotation_id = 1

    for image_file in tqdm(images_dir.glob("*.*"), desc="Converting images"):
        image_id = int(image_file.stem)
        image_info = {
            "id": image_id,
            "file_name": image_file.name,
            "width": None,  # Placeholder, should be filled with actual width
            "height": None,  # Placeholder, should be filled with actual height
        }
        coco_format["images"].append(image_info)

        label_file = (labels_dir / image_file.stem).with_suffix(".txt")
        if label_file.exists():
            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_idx, x_center, y_center, width, height = map(float, parts)
                bbox = [
                    x_center - width / 2,
                    y_center - height / 2,
                    width,
                    height,
                ]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_idx),
                    "bbox": bbox,
                    "area": width * height,
                    "iscrowd": 0,
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

    # Save COCO JSON
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    logger.info(f"Dataset converted to COCO format and saved to {output_json}")


def convert_coco_to_yolo(coco_json, output_dir):
    """
    Converts a COCO dataset to YOLOv5 format.

    Args:
        coco_json (str): Path to the COCO JSON file.
        output_dir (str): Path to the output YOLOv5 dataset directory.

    Raises:
        FileNotFoundError: If the COCO JSON file does not exist.
    """
    coco_json = Path(coco_json)
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON file not found: {coco_json}")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco_data = json.load(f)

    # Create a mapping of category IDs to names
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Save class names to data.yaml
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({"names": list(categories.values())}, f)

    # Determine the base directory for images
    coco_images_dir = coco_json.parent / "images"

    # Process images and annotations
    for image in tqdm(coco_data["images"], desc="Processing images"):
        image_id = image["id"]
        image_name = image["file_name"]
        source_image_path = coco_images_dir / image_name

        if not source_image_path.exists():
            logger.error(f"Image file not found: {source_image_path}")
            continue

        shutil.copy(source_image_path, images_dir / image_name)

        # Collect annotations for this image
        annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]

        label_file = labels_dir / f"{Path(image_name).stem}.txt"
        with open(label_file, "w") as f:
            for ann in annotations:
                category_id = ann["category_id"]
                bbox = ann["bbox"]  # [x_min, y_min, width, height]
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                width = bbox[2]
                height = bbox[3]

                # Write YOLO format: class_id x_center y_center width height
                class_id = list(categories.keys()).index(category_id)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    logger.info(f"Dataset converted to YOLOv5 format and saved to {output_dir}")
