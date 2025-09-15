# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

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
    parser.add_argument("--fig_path", type=Path, help="Path to store output figures.")
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Folder to save the results. If not provided, data is not saved.",
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Show result in interactive window"
    )
    args = parser.parse_args()
    return Args(**vars(args))


def main(args: Args):
    variables = ("timestamp", "diameter_3d", "confidence")
    file_path = args.data_path / f"{args.participant_id}.hdf5"
    hdf_path_info = {"group": "trials", "topic": "pupil"}
    participant_data, participant_metadata = d_import.get_resampled_participant_data(
        file_path, variables=variables, **hdf_path_info
    )
    p_data = da.convert_to_array(participant_data)
    max_values = dict.fromkeys(p_data.keys())
    d_trendlines = dict.fromkeys(p_data.keys())
    for task, task_data in p_data.items():
        da.normalize_pupil_diameter(task_data)
        da.remove_low_confidence(task_data)
        max_values[task] = da.get_max_values(task_data["diameter_3d"])
        d_trendlines[task] = da.get_trendlines_by_task(task_data["diameter_3d"])
    t = {
        key: np.nanmax(array["timestamp"], axis=(1, 2)) for key, array in p_data.items()
    }
    d_plot.plot_trendlines(
        t,
        d_trendlines,
        title="Fractional Change in Pupil Diameter from Baseline Comparison for Two Tasks",
    )
    d_plot.plot_max_values(max_values)
    if args.show_plot:
        plt.show()
    pass


if __name__ == "__main__":
    args = get_args()
    main(args)
