"""
This module provides functionality to download common object detection datasets.

Functions:
    download_coco(output_dir: str):
        Downloads the COCO dataset to the specified directory.
    download_voc(output_dir: str):
        Downloads the Pascal VOC dataset to the specified directory.
"""

import os
import requests
import zipfile
from lgg import logger
from tqdm import tqdm


def download_coco(output_dir: str):
    """
    Downloads the COCO dataset to the specified directory.

    Args:
        output_dir (str): Path to the directory where the dataset will be downloaded.

    Returns:
        None
    """
    # COCO dataset URLs (2017)
    COCO_URLS = {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "test_images": "http://images.cocodataset.org/zips/test2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    output_paths = {
        key: os.path.join(output_dir, f"{key}.zip") for key in COCO_URLS.keys()
    }

    os.makedirs(output_dir, exist_ok=True)

    for key, coco_url in COCO_URLS.items():
        logger.info(f"Downloading {key}...")
        response = requests.get(coco_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(output_paths[key], "wb") as f,
            tqdm(
                desc=f"Downloading {key}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=5 * 1024 * 1024,  # 5 MB
            ) as progress_bar,
        ):
            for chunk in response.iter_content(chunk_size=5 * 1024 * 1024):  # 5 MB
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        logger.info(f"Extracting {key}...")
        with zipfile.ZipFile(output_paths[key], "r") as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(output_paths[key])
    logger.info(f"COCO dataset downloaded and extracted successfully to {output_dir}.")


def download_voc(output_dir: str):
    """
    Downloads the Pascal VOC dataset to the specified directory.

    Args:
        output_dir (str): Path to the directory where the dataset will be downloaded.

    Returns:
        None
    """
    voc_url = (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    )
    output_path = os.path.join(output_dir, "VOCtrainval_11-May-2012.tar")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Downloading Pascal VOC dataset...")
    response = requests.get(voc_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(output_path, "wb") as f,
        tqdm(
            desc="Downloading VOC dataset",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=5 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    logger.info("Extracting Pascal VOC dataset...")
    os.system(f'tar -xvf "{output_path}" -C "{output_dir}"')

    os.remove(output_path)
    logger.info("Pascal VOC dataset downloaded and extracted successfully.")
