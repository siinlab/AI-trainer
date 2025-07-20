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
