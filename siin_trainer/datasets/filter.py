"""
This module provides functionality to filter out specific objects from a dataset.

Functions:
- discard_objects: Removes specified objects from a dataset by updating label files and optionally creating a new dataset.
"""

from pathlib import Path
import yaml
from tqdm import tqdm
from lgg import logger
import shutil


def discard_objects(dataset_path, objects_to_discard, output_dir=None):
    """
    Discards specified objects from a dataset by removing their entries in label files.

    Args:
        dataset_path (str): Path to the dataset directory containing 'images' and 'labels'.
        objects_to_discard (list): List of object names to discard.
        output_dir (str, optional): Path to the output directory. If None, modifies the dataset in place.

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

    # Identify class indices to discard
    indices_to_discard = [
        class_names.index(obj) for obj in objects_to_discard if obj in class_names
    ]

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir = output_dir / "labels"
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to the new dataset
        for image_file in images_dir.glob("*.*"):
            shutil.copy(image_file, output_images_dir / image_file.name)

        # Update data.yaml
        updated_class_names = [
            class_name
            for class_name in class_names
            if class_name not in objects_to_discard
        ]
        updated_data_yaml = output_dir / "data.yaml"
        with open(updated_data_yaml, "w") as f:
            yaml.dump({"names": updated_class_names}, f)
    else:
        output_labels_dir = labels_dir

    # Process label files
    for label_file in tqdm(labels_dir.glob("*.txt"), desc="Processing labels"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_idx = int(parts[0])
                if class_idx not in indices_to_discard:
                    updated_lines.append(line)

        # Write updated label file
        output_label_file = output_labels_dir / label_file.name
        with open(output_label_file, "w") as f:
            f.writelines(updated_lines)

    logger.info(
        f"Objects discarded successfully. Updated labels and images saved to {output_dir if output_dir else labels_dir}"
    )
