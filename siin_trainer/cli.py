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
        logger.error(f"ValueError: {e}")
        click.echo(f"Error: {e}", err=True)
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"Unexpected error: {e}", err=True)


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
        logger.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


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
        logger.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


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
        click.echo("Filtering completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"Unexpected error: {e}", err=True)


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
        logger.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


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
        logger.error(f"FileNotFoundError: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
