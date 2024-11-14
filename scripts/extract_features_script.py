from argparse import ArgumentParser
from pathlib import Path
import h5py


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the HDF file containing processed data."
    )
    parser.add_argument(
        "export_path", type=Path, help="Path to directory to save the feature data."
    )
    return parser.parse_args()


def main(data_filepath: Path, export_path: Path):
    with h5py.File(data_filepath) as f_root:
        pass


if __name__=="__main__":
    args = get_args()
    main(**vars(args))