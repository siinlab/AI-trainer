"""
This module provides functionality to visualize random samples from a dataset.

Functions:
- visualize_samples: Visualizes N random samples from a dataset and saves them to a specified directory.
"""

import random
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm
from lgg import logger


def visualize_samples(dataset_path, output_dir, num_samples=5):
    """
    Visualizes N random samples from the dataset and saves them in a directory.

    Args:
        dataset_path (str): Path to the dataset directory containing 'images' and 'labels'.
        output_dir (str): Path to the directory where visualized samples will be saved.
        num_samples (int): Number of random samples to visualize. Defaults to 5.

    Raises:
        FileNotFoundError: If the dataset does not contain 'images' and 'labels' directories.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "Dataset must contain 'images' and 'labels' directories."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class names from data.yaml
    with open(dataset_path / "data.yaml", "r") as f:
        data = yaml.safe_load(f)
        class_names = data.get("names", [])

    image_paths = list(images_dir.glob("*"))
    selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))

    for image_path in tqdm(selected_images, desc="Visualizing samples"):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            continue

        # Read corresponding label file
        label_path = (labels_dir / image_path.stem).with_suffix(".txt")
        if label_path.exists():
            with open(label_path, "r") as label_file:
                lines = label_file.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_idx, x_center, y_center, width, height = map(float, parts)
                class_idx = int(class_idx)

                # Convert YOLO format to bounding box coordinates
                img_h, img_w, _ = image.shape
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)

                # Draw bounding box and label
                color = (0, 255, 0)  # Green color for bounding box
                label = (
                    class_names[class_idx]
                    if class_idx < len(class_names)
                    else str(class_idx)
                )
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Save visualized image
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), image)

    logger.info(f"Visualized samples saved to {output_dir}")
