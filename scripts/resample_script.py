import pupiltools.data_import as d_import
import pupiltools.data_plotting as d_plot
import pupiltools.data_analysis as da
import pupiltools.export as pt_export
from pupiltools.utilities import save_figure

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path


@dataclass
class Args:
    participant_id: str
    data_path: Path
    fig_path: Path
    T_resample: float
    show_plot: bool
    resampled_data_path: Path


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("participant_id", help="ID code of participant (ex. 'P04').")
    parser.add_argument(
        "data_path", type=Path, help="Path to directory containing the HDF files."
    )
    parser.add_argument(
        "T_resample", type=float, help="Sample time interval to create from resample."
    )
    parser.add_argument(
        "--fig_path", type=Path, default=None, help="Path to store output figures."
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Show result in interactive window"
    )
    parser.add_argument(
        "--resampled_data_path", type=Path, default=None,
        help="Folder to save the resampled data. If not provided, data is not saved."
    )
    args = parser.parse_args()
    return Args(**vars(args))


if __name__ == "__main__":
    args = get_args()
    variables = "all"
    eyes = (0, 1)
    file_path = args.data_path / f"{args.participant_id}.hdf5"
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_data, participant_metadata = d_import.get_raw_participant_data(
        file_path, variables=variables, **hdf_path_info
    )
    resampled_data = da.resample_data(participant_data, args.T_resample)
    resample_fig = d_plot.resample_comparison(
        participant_data[0]["data"][0],
        resampled_data[0]["data"][:,0],
        "diameter_3d",
        "Diameter [mm]"
    )
    if args.resampled_data_path is not None:
        complete_data_structure = {
            "attributes": participant_metadata,
            "data": resampled_data
        }
        if not args.resampled_data_path.exists():
            args.resampled_data_path.mkdir()
        export_path = args.resampled_data_path / f"{args.participant_id}.hdf5"
        pt_export.export_hdf(export_path, complete_data_structure)
    if args.fig_path is not None:
        figname = f"resample-comparison-{args.participant_id}-dt-{args.T_resample*1000:0.0f}ms"
        save_figure(resample_fig, args.fig_path, figname, ("png", "svg"))
    if args.show_plot:
        plt.show()
