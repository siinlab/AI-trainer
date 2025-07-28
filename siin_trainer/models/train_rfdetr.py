"""
This module provides functionality to train RF-DETR models on custom datasets.

Functions:
    train_rfdetr_model(data_path: str, model_name: str, epochs: int, batch_size: int, device: str, resume: str):
        Trains an RF-DETR model using the specified parameters.
"""

from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETRLarge, RFDETRBase


def train_rfdetr_model(
    data_path: str,
    model_name: str = "RFDETRMedium",
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    resume: str = None,
):
    """
    Train an RF-DETR model on a custom dataset.

    Args:
        data_path (str): Path to the dataset YAML file.
        model_name (str): Name of the RF-DETR model to use (e.g., 'RFDETRMedium', 'RFDETRNano', 'RFDETRSmall', 'RFDETRLarge', 'RFDETRBase').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to use for training ('cuda' or 'cpu').
        resume (str): Path to the checkpoint file to resume training from.

    Returns:
        None
    """
    # Initialize the RF-DETR model
    if model_name == "RFDETRMedium":
        model = RFDETRMedium()
    elif model_name == "RFDETRNano":
        model = RFDETRNano()
    elif model_name == "RFDETRSmall":
        model = RFDETRSmall()
    elif model_name == "RFDETRLarge":
        model = RFDETRLarge()
    elif model_name == "RFDETRBase":
        model = RFDETRBase()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Train the model
    model.train(
        dataset_dir=data_path,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        resume=resume,
        num_workers=8,
        lr=1e-4,
        grad_accum_steps=4,
        wandb=True,
        project="AI-Trainer-RFDETR-Runs",
    )

    # Export the trained model to ONNX format
    model.export(output_dir="AI-Trainer-RFDETR-Runs", simplify=True, opset_version=12)
