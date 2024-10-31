import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot
import pupiltools.data_analysis as da
from pupiltools.utilities import save_figure, make_digit_str

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


@dataclass
class Args:
    data_path: Path
    fig_path: Path
    show_plot: bool


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "data_path", type=Path, help="Path to directory containing the HDF files."
    )
    parser.add_argument(
        "--fig_path", type=Path, help="Path to store output figures."
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Show result in interactive window"
    )
    args = parser.parse_args()
    return Args(**vars(args))


def main(args: Args):
    variables = ("timestamp", "diameter_3d", "confidence")
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_ids = ["P"+make_digit_str(i, 2) for i in range(1,31) if i not in (18, 20)]
    max_val_hist = np.zeros((20,2), dtype=np.int64)
    for participant_id in participant_ids:
        print(participant_id)
        file_path = args.data_path / f"{participant_id}.hdf5"
        participant_data, _ = d_import.get_resampled_participant_data(
            file_path, variables=variables, **hdf_path_info
        )
        p_data_array = da.convert_to_array(participant_data)
        da.normalize_pupil_diameter(p_data_array)
        da.remove_low_confidence(p_data_array)
        max_values = da.get_max_values(p_data_array["diameter_3d"])
        for eye in (0,1):
            for task in (0,1):
                max_val_hist_vals, bin_edges = np.histogram(
                    max_values[:,eye,task], bins=20, range=(0,0.5)
                )
                max_val_hist[:,task] += max_val_hist_vals
    #d_plot.plot_max_values(max_values)
    print(np.sum(max_val_hist, axis=0))
    d_plot.manual_hist(max_val_hist, bin_edges)
    if args.show_plot:
        plt.show()
    pass


if __name__=="__main__":
    args = get_args()
    main(args)