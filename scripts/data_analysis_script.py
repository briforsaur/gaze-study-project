import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot
import pupiltools.data_analysis as da

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path



@dataclass
class Args:
    participant_id: str
    data_path: Path
    td_analysis: bool


def time_delta_analysis(participant_data):
    t_deltas = da.get_time_deltas(participant_data)
    td_stats = da.get_time_delta_stats(t_deltas)
    stats = (f"\nMean: {td_stats['mean']:.5f}\n"
             f"Median: {td_stats['median']:.5f}\n"
             f"95th percentile: {td_stats['95th percentile']:.5f}\n"
             f"99th percentile: {td_stats['99th percentile']:.5f}\n"
             f"Max: {td_stats['max']:.5f}\n"
            )
    print(stats)
    td_fig = d_plot.plot_dt_histogram(t_deltas)
    return td_fig


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("participant_id", help="ID code of participant (ex. 'P04').")
    parser.add_argument("data_path", type=Path, help="Path to directory containing the HDF files.")
    parser.add_argument("--td_analysis", action="store_true", help="Perform time delta analysis")
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
    if args.td_analysis:
        td_fig = time_delta_analysis(participant_data)
    plt.show()