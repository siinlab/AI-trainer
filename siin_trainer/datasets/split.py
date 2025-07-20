"""
This module provides functionality to split a dataset into training, validation, and test sets.

The dataset is expected to have the following structure:
- images/: Directory containing image files.
- labels/: Directory containing corresponding label files.

The module includes:
- A function `split_dataset` to perform the dataset splitting.
- A `main` function to handle command-line arguments and invoke the splitting logic.
"""

from lgg import logger


def split_dataset(dataset_path, val_ratio, test_ratio, seed=1):
    """Splits a dataset into training, validation, and test sets.

    The dataset should be Yolov5-formatted, ie. it should contain two directories: 'images' and 'labels'.

    Args:
        dataset_path (str): Path to the dataset directory.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int): Random seed for reproducibility. Defaults to 1.

    Raises:
        ValueError: If the sum of val_ratio and test_ratio is >= 1.
        FileNotFoundError: If the dataset does not contain 'images' and 'labels' directories.

    The function creates three text files in the dataset directory:
    - train.txt: Contains paths to training images.
    - val.txt: Contains paths to validation images (if any).
    - test.txt: Contains paths to test images (if any).
    """
    from pathlib import Path
    import random

    dataset = Path(dataset_path)

    # check if val + test is less than 1
    if val_ratio + test_ratio >= 1:
        raise ValueError("val + test should be less than 1")

    # check if dataset exists and contains two directories: images and labels
    if not (dataset / "images").exists() or not (dataset / "labels").exists():
        raise FileNotFoundError("images and labels directories not found in dataset")

    # get image and label paths
    img_paths = list((dataset / "images").glob("*"))
    logger.info(f"Found {len(img_paths)} images in `{dataset}`")

    # seed the generator
    random.seed(seed)

    # shuffle the dataset
    random.shuffle(img_paths)

    # Compute the dataset splits
    n = len(img_paths)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    # print the number of images in each set
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Split dataset
    train = img_paths[:n_train]
    val = img_paths[n_train : n_train + n_val]
    test = img_paths[n_train + n_val :]

    # Save image paths to text files
    with open(dataset / "train.txt", "w") as f:
        f.write("\n".join(map(str, train)))

    if len(val) > 0:
        with open(dataset / "val.txt", "w") as f:
            f.write("\n".join(map(str, val)))

    if len(test) > 0:
        with open(dataset / "test.txt", "w") as f:
            f.write("\n".join(map(str, test)))

    logger.info("Dataset split successfully.")
