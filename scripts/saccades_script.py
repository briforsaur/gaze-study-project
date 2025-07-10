from argparse import ArgumentParser
from dataclasses import dataclass
import h5py
import logging
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pupiltools.constants import TASK_TYPES

logger = logging.getLogger(__name__)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

@dataclass
class Args:
    data_path: Path
    export_path: Path
    comparison_file: Path


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "data_path", type=Path, help="Path to HDF file of data to plot."
    )
    parser.add_argument(
        "export_path", type=Path, help="Path to directory to save the script outputs."
    )
    parser.add_argument(
        "--comparison_file", type=Path, help="Path to HDF file of data to add for comparison.", default=None
    )
    return Args(**vars(parser.parse_args()))


def main(data_path: Path, export_path: Path, comparison_file: Path = None): #type: ignore
    participant = "P05"
    variables = ("timestamp", "theta", "phi")
    task_number = 35
    if comparison_file is not None:
        files = (data_path, comparison_file)
        legend = ("Unfiltered", "Filtered")
    else:
        files = (data_path,)
        legend = ()
    w = 12
    fig = plt.figure(figsize=(w, w), dpi=300)
    axs = fig.subplots(2, 1)
    for file in files:
        task_data = {}
        with h5py.File(file, mode='r') as f_root:
            for task, ax in zip(TASK_TYPES, axs):
                assert isinstance(ax, matplotlib.axes.Axes)
                dataset = f_root[f"/{participant}/{task}"]
                assert isinstance(dataset, h5py.Dataset)
                track_data = dataset.fields(variables)[:]
                assert isinstance(track_data, np.ndarray)
                plot_data = track_data[:, task_number, :]
                ax.plot(plot_data['timestamp'][:,0], plot_data['theta'][:,0])
                ax.set_title(task)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Polar Angle, $\\theta$ [rad]")
                ax.legend(legend)
    if not export_path.exists():
        export_path.mkdir(parents=True)
    fig.savefig(export_path / f"{participant}_saccade.pdf", bbox_inches="tight")
    plt.show()

    
    



if __name__=="__main__":
    args = get_args()
    main(args.data_path, args.export_path, args.comparison_file)
