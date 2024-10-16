import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path



@dataclass
class Args:
    participant_id: str
    data_path: Path


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("participant_id", help="ID code of participant (ex. 'P04').")
    parser.add_argument("data_path", type=Path, help="Path to directory containing the HDF files.")
    args = parser.parse_args()
    return Args(**vars(args))

if __name__ == "__main__":
    args = get_args()
    variables = ("timestamp", "diameter_3d")
    eyes = (0, 1)
    file_path = args.data_path / f"{args.participant_id}.hdf5"
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_data = d_import.get_raw_participant_data(file_path, variables=variables, **hdf_path_info)
    fig = d_plot.plot_raw_pupil_diameter_comparison(participant_data)
    plt.show()