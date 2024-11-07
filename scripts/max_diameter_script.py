import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot
import pupiltools.data_analysis as da
from pupiltools.utilities import save_figure, make_digit_str

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import logging
logger = logging.getLogger(__name__)


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
    logging.basicConfig(level=logging.INFO)
    variables = ("timestamp", "diameter_3d", "confidence")
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_ids = ["P"+make_digit_str(i, 2) for i in range(1,31)]
    n_bins = 20
    max_val_hist = np.zeros((n_bins,2), dtype=np.int64)
    for participant_id in participant_ids:
        logger.info(f"Processing {participant_id}")
        file_path = args.data_path / f"{participant_id}.hdf5"
        participant_data, _ = d_import.get_resampled_participant_data(
            file_path, variables=variables, **hdf_path_info
        )
        p_data_arraydict = da.convert_to_array(participant_data)
        max_values = {}
        for task, p_data_array in p_data_arraydict.items():
            da.remove_low_confidence(p_data_array)
            da.interpolate_nan(p_data_array)
            da.normalize_pupil_diameter(p_data_array)
            max_values.update({task: da.get_max_values(p_data_array["diameter_3d"])})
            if np.any(np.isnan(max_values[task])):
                logger.warning(f"All NaNs for {participant_id}, task '{task}'")
        for eye in (0,1):
            for i, task in enumerate(max_values.keys()):
                max_val_hist_vals, bin_edges = np.histogram(
                    max_values[task][:,eye], bins=n_bins, range=(0,0.5)
                )
                max_val_hist[:,i] += max_val_hist_vals
    # Find pupil diameter increase that maximally separates the classes
    split = da.calc_split(max_val_hist, bin_edges)
    print(f"Split: {split}")
    d_plot.manual_hist(max_val_hist, bin_edges, split, normalize=True)
    if args.show_plot:
        plt.show()
    pass


if __name__=="__main__":
    args = get_args()
    main(args)