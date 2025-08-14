from lgg import logger
import click


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Entry point for the Siin Trainer CLI."""
    if ctx.invoked_subcommand is None:
        logger.info("Welcome to Siin Trainer CLI!")
    else:
        logger.info(f"Running subcommand: {ctx.invoked_subcommand}")


@main.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the dataset directory.",
)
@click.option(
    "--val",
    type=float,
    default=0.15,
    help="Proportion of the dataset to use for validation.",
)
@click.option(
    "--test",
    type=float,
    default=0.05,
    help="Proportion of the dataset to use for testing.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def split_dataset(dataset, val, test, seed):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
        dataset (str): Path to the dataset directory.
        val (float): Proportion of the dataset to use for validation. Defaults to 0.15.
        test (float): Proportion of the dataset to use for testing. Defaults to 0.05.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If the sum of val and test is >= 1.
        FileNotFoundError: If the dataset does not contain 'images' and 'labels' directories.
    """
    from .datasets.split import split_dataset as split_logic

    try:
        split_logic(dataset, val, test, seed)
    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the output directory for the merged dataset.",
)
@click.option(
    "--datasets",
    type=click.Path(exists=True, file_okay=False),
    multiple=True,
    required=True,
    help="Paths to the YOLO datasets to merge. Provide multiple paths separated by spaces.",
)
def merge_datasets(output, datasets):
    """
    Merges multiple YOLO datasets into a single dataset.

    Args:
        output (str): Path to the output directory for the merged dataset.
        datasets (tuple): Paths to the YOLO datasets to merge.

    Raises:
        FileNotFoundError: If any of the dataset paths do not exist or are not properly formatted.
    """
    from .datasets.merge import merge_yolo_datasets

    try:
        merge_yolo_datasets(output, *datasets)
        logger.info("Datasets merged successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the dataset directory containing 'images' and 'labels'.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the directory where visualized samples will be saved.",
)
@click.option(
    "--num-samples",
    type=int,
    default=5,
    help="Number of random samples to visualize. Defaults to 5.",
)
def visualize_dataset(dataset, output, num_samples):
    """
    Visualizes N random samples from the dataset and saves them in a directory.

    Args:
        dataset (str): Path to the dataset directory containing 'images' and 'labels'.
        output (str): Path to the directory where visualized samples will be saved.
        num_samples (int): Number of random samples to visualize. Defaults to 5.

    Raises:
        FileNotFoundError: If the dataset does not contain 'images' and 'labels' directories.
    """
    from .datasets.visualize import visualize_samples

    try:
        visualize_samples(dataset, output, num_samples)
        logger.info("Visualization completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the dataset directory containing 'images' and 'labels'.",
)
@click.option(
    "--objects",
    type=str,
    multiple=True,
    required=True,
    help="Names of objects to discard. Provide multiple names separated by spaces.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the output directory. If not provided, modifies the dataset in place.",
)
def filter_objects(dataset, objects, output):
    """
    Discards specified objects from a dataset by removing their entries in label files.

    Args:
        dataset (str): Path to the dataset directory containing 'images' and 'labels'.
        objects (tuple): Names of objects to discard.
        output (str, optional): Path to the output directory. If None, modifies the dataset in place.

    Raises:
        FileNotFoundError: If the dataset does not contain 'images', 'labels', or 'data.yaml'.
    """
    from .datasets.filter import discard_objects

    try:
        discard_objects(dataset, objects, output)
        logger.info("Filtering completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the YOLOv5 dataset directory containing 'images' and 'labels'.",
)
@click.option(
    "--output-json",
    type=click.Path(writable=True),
    required=True,
    help="Path to the output COCO JSON file.",
)
def yolo_to_coco(dataset, output_json):
    """
    Converts a YOLOv5 dataset to COCO format.

    Args:
        dataset (str): Path to the YOLOv5 dataset directory containing 'images' and 'labels'.
        output_json (str): Path to the output COCO JSON file.

    Raises:
        FileNotFoundError: If the dataset does not contain 'images', 'labels', or 'data.yaml'.
    """
    from .datasets.convert import convert_yolo_to_coco

    try:
        convert_yolo_to_coco(dataset, output_json)
        logger.info("Conversion to COCO format completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--coco-json",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Path to the COCO JSON file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the output YOLOv5 dataset directory.",
)
def coco_to_yolo(coco_json, output_dir):
    """
    Converts a COCO dataset to YOLOv5 format.

    Args:
        coco_json (str): Path to the COCO JSON file.
        output_dir (str): Path to the output YOLOv5 dataset directory.

    Raises:
        FileNotFoundError: If the COCO JSON file does not exist.
    """
    from .datasets.convert import convert_coco_to_yolo

    try:
        convert_coco_to_yolo(coco_json, output_dir)
        logger.info("Conversion to YOLOv5 format completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--data",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Path to the dataset YAML file.",
)
@click.option(
    "--model",
    type=str,
    default="yolov8n",
    help="Name of the YOLO model to use (e.g., 'yolov8n', 'yolov8s').",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs. Defaults to 50.",
)
@click.option(
    "--img-size",
    type=int,
    default=640,
    help="Image size for training. Defaults to 640.",
)
@click.option(
    "--batch",
    type=int,
    default=16,
    help="Batch size for training. Defaults to 16.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for training (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.",
)
@click.option(
    "--cache",
    type=str,
    default="ram",
    help="Cache type to use during training. Defaults to 'ram'.",
)
def train_ultralytics(data, model, epochs, img_size, batch, device, cache):
    """
    Trains a YOLO model on a custom dataset.

    Args:
        data (str): Path to the dataset YAML file.
        model (str): Name of the YOLO model to use.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        batch (int): Batch size for training.
        device (str): Device to use for training.
        cache (str): Cache type to use during training.

    Raises:
        FileNotFoundError: If the dataset YAML file does not exist.
    """
    from .models.train_ultralytics import train_ultralytics_model

    try:
        train_ultralytics_model(
            data_path=data,
            model_name=model,
            epochs=epochs,
            img_size=img_size,
            batch=batch,
            device=device,
            cache=cache,
        )
        logger.info("Model training completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--name",
    type=click.Choice(["coco", "voc"], case_sensitive=False),
    required=False,
    help="Name of the dataset to download (e.g., 'coco', 'voc').",
)
@click.option(
    "--url",
    type=str,
    required=False,
    help="Custom URL of the dataset to download.",
)
@click.option(
    "--dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the directory where the dataset will be saved.",
)
def download_dataset(name, url, dir):
    """
    Downloads a specified dataset or a dataset from a custom URL and saves it to the given directory.

    Args:
        name (str): Name of the dataset to download (e.g., 'coco', 'voc').
        url (str): Custom URL of the dataset to download.
        dir (str): Path to the directory where the dataset will be saved.

    Raises:
        ValueError: If neither name nor URL is provided.
    """
    from .datasets.download import download_coco, download_voc, download_from_url

    try:
        if name:
            if name.lower() == "coco":
                download_coco(dir)
            elif name.lower() == "voc":
                download_voc(dir)
            else:
                raise ValueError(f"Unsupported dataset: {name}")
        elif url:
            download_from_url(url, dir)
        else:
            raise ValueError("Either --name or --url must be provided.")

        logger.info("Dataset downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}", exc_info=True)


@main.command()
@click.option(
    "--data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the dataset directory.",
)
@click.option(
    "--model",
    type=click.Choice(
        ["RFDETRMedium", "RFDETRNano", "RFDETRSmall", "RFDETRLarge", "RFDETRBase"],
        case_sensitive=True,
    ),
    default="RFDETRMedium",
    help="Name of the RF-DETR model to use.",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs. Defaults to 50.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for training. Defaults to 16.",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for training (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, file_okay=True),
    required=False,
    help="Path to the checkpoint file to resume training from.",
)
def train_rfdetr(data, model, epochs, batch_size, device, resume):
    """
    Trains an RF-DETR model on a custom dataset.

    Args:
        data (str): Path to the dataset directory.
        model (str): Name of the RF-DETR model to use.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to use for training.
        resume (str): Path to the checkpoint file to resume training from.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    from .models.train_rfdetr import train_rfdetr_model

    try:
        train_rfdetr_model(
            data_path=data,
            model_name=model,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            resume=resume,
        )
        logger.info("RF-DETR model training completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


@main.command()
@click.option(
    "--video",
    type=click.Path(exists=True, file_okay=True),
    required=True,
    help="Path to the video file from which to extract frames.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Path to the directory where extracted frames will be saved.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.5,
    help="Threshold for similarity when processing frames. Defaults to 0.5.",
)
def extract_frames(video, output_dir, similarity_threshold):
    """
    Extracts frames from a video file and saves them as images in the specified directory.

    Args:
        video (str): Path to the video file.
        output_dir (str): Path to the directory where extracted frames will be saved.
        similarity_threshold (float): Threshold for similarity when processing frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    from .datasets.extract_images import extract_frames_from_video

    try:
        extract_frames_from_video(video, output_dir, similarity_threshold)
        logger.info("Frames extracted successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
