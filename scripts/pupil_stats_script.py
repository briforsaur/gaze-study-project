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
    variables = ("timestamp", "diameter_3d")
    file_path = args.data_path / f"{args.participant_id}.hdf5"
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_data, participant_metadata = d_import.get_resampled_participant_data(
        file_path, variables=variables, **hdf_path_info
    )
    trendline_array = da.get_trendlines_by_task(participant_data)
    t_max = trendline_array.shape[0]*0.01
    t = np.linspace(0, t_max, trendline_array.shape[0])
    fig, axs = plt.subplots(2)
    task_labels = ["action", "observation"]
    for i, ax in enumerate(axs):
        ax.set_title(f"eye{i}")
        for j in (0, 1):
            ax.plot(t, trendline_array[:,i,j], label=task_labels[j])
        ax.legend()
    plt.show()
    pass


if __name__=="__main__":
    args = get_args()
    main(args)