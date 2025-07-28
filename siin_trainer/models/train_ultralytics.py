"""
This module provides functionality to train Ultralytics YOLO models on custom datasets.

Functions:
    train_ultralytics_model(data_path: str, model_name: str, epochs: int, img_size: int, batch: int, device: str, cache: str):
        Trains a YOLO model using the specified parameters.
"""

from ultralytics import YOLO


def train_ultralytics_model(
    data_path: str,
    model_name: str = "yolov8n",
    epochs: int = 50,
    img_size: int = 640,
    batch=16,
    device="cuda",
    cache="ram",
):
    """
    Train an Ultralytics YOLO model on a custom dataset.

    Args:
        data_path (str): Path to the dataset YAML file.
        model_name (str): Name of the YOLO model to use (e.g., 'yolov8n', 'yolov8s', 'yolo11.yaml').
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        batch (int): Batch size for training.
        device (str): Device to use for training ('cuda' or 'cpu').
        cache (str): Cache mode ('ram' or 'disk').

    Returns:
        None
    """
    # Initialize the YOLO model
    model = YOLO(model_name)

    # Train the model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        cache=cache,
        project="AI-Trainer-Ultralytics-Runs",
    )
