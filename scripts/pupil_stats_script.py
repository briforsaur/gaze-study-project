import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot
import pupiltools.data_analysis as da
from pupiltools.utilities import save_figure

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


@dataclass
class Args:
    participant_id: str
    data_path: Path
    fig_path: Path
    save_path: Path
    show_plot: bool


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("participant_id", help="ID code of participant (ex. 'P04').")
    parser.add_argument(
        "data_path", type=Path, help="Path to directory containing the HDF files."
    )
    parser.add_argument(
        "--fig_path", type=Path, help="Path to store output figures."
    )
    parser.add_argument(
        "--save_path", type=Path, default=None,
        help="Folder to save the results. If not provided, data is not saved."
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Show result in interactive window"
    )
    args = parser.parse_args()
    return Args(**vars(args))


def main(args: Args):
    variables = ("timestamp", "diameter_3d", "confidence")
    file_path = args.data_path / f"{args.participant_id}.hdf5"
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_data, participant_metadata = d_import.get_resampled_participant_data(
        file_path, variables=variables, **hdf_path_info
    )
    p_data_array = da.convert_to_array(participant_data)
    da.normalize_pupil_diameter(p_data_array)
    da.remove_low_confidence(p_data_array)
    d_trendline_array = da.get_trendlines_by_task(p_data_array["diameter_3d"])
    d_plot.plot_trendlines(d_trendline_array)
    if args.show_plot:
        plt.show()
    pass


if __name__=="__main__":
    args = get_args()
    main(args)