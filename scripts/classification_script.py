from argparse import ArgumentParser
from pathlib import Path
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the NPZ file containing feature data."
    )
    parser.add_argument(
        "results_path", type=Path, help="Path to directory to save the results."
    )
    return parser.parse_args()


def main(data_filepath: Path, results_path: Path):
    # Load feature data file
    # For all participants
        # Split data into training and validation (leave one out)
        # Split data into features and class labels
        # Train model
        # Get final training metrics
        # Test model on left out data
        # Get final test metrics
    pass


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
