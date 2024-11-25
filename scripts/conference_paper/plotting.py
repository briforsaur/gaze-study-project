from argparse import ArgumentParser
import h5py
from pathlib import Path

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the HDF file containing processed data."
    )
    return parser.parse_args()

def main(data_filepath: Path):
    participant_id = "P23"
    tasks = ("action", "observation")
    variables = ("timestamp", "diameter_3d")
    diameter = {}
    with h5py.File(data_filepath, mode='r') as f_root:
        for task in tasks:
            dataset = f_root["/" + "/".join((participant_id, task))]
            diameter.update({task: dataset.fields(variables)[:]})


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))