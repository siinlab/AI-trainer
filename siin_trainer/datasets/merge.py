"""
This script merges two or more YOLO object detection datasets while preserving the same classes and creating new classes when needed.

Each dataset should have the following structure:
- images/: Directory containing image files.
- labels/: Directory containing corresponding label files in YOLO format.
- data.yaml: File containing class definitions.

The script will:
- Combine images and labels into a single dataset.
- Merge class definitions, ensuring no duplicates.
- Update label files to reflect new class indices if classes are added.
"""

import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
from lgg import logger


def merge_yolo_datasets(output_dir, *dataset_paths):
    """
    Merges multiple YOLO datasets into a single dataset.

    Args:
        output_dir (str): Path to the output directory for the merged dataset.
        *dataset_paths (str): Paths to the YOLO datasets to merge.

    Raises:
        FileNotFoundError: If any of the dataset paths do not exist or are not properly formatted.
    """
    output_dir = Path(output_dir).resolve()
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_yaml = output_dir / "data.yaml"

    # Create output directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    merged_classes = []
    image_counter = 0

    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if (
            not (dataset_path / "images").exists()
            or not (dataset_path / "labels").exists()
            or not (dataset_path / "data.yaml").exists()
        ):
            raise FileNotFoundError(
                f"Dataset at {dataset_path} is missing required directories or files."
            )

        # Load class definitions
        with open(dataset_path / "data.yaml", "r") as f:
            data = yaml.safe_load(f)
            classes = data.get("names", [])

        # Merge class definitions
        for cls in classes:
            if cls not in merged_classes:
                merged_classes.append(cls)

        # Copy images and labels
        for image_path in tqdm(
            sorted((dataset_path / "images").glob("*")),
            desc=f"Processing images from {dataset_path}",
        ):
            new_image_name = f"{image_counter:06d}{image_path.suffix}"
            shutil.copy(image_path, output_images / new_image_name)

            # Copy corresponding label file
            label_path = (dataset_path / "labels" / image_path.name).with_suffix(".txt")
            if label_path.exists():
                with open(label_path, "r") as label_file:
                    lines = label_file.readlines()

                # Update class indices in label file
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_idx = int(parts[0])
                        new_class_idx = merged_classes.index(classes[class_idx])
                        updated_lines.append(" ".join([str(new_class_idx)] + parts[1:]))

                with open(
                    output_labels / f"{image_counter:06d}.txt", "w"
                ) as new_label_file:
                    new_label_file.write("\n".join(updated_lines))

            image_counter += 1

    # Write merged class definitions to data.yaml
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"names": merged_classes, "nc": len(merged_classes), "path": "./"}, f, allow_unicode=True)

    logger.info(f"Merged dataset created at {output_dir}")
