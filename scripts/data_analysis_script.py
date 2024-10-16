import pupiltools.data_import as di

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
import numpy as np
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
    participant_data = di.get_raw_participant_data(file_path, variables=variables, **hdf_path_info)
    fig, axs = plt.subplots(2,1)
    tasks = ("action", "observation")
    plot_topics: dict[str, mpl_axes] = dict(zip(tasks, axs))
    for title, ax in plot_topics.items():
        ax.set_title(title.capitalize())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Normalized Diameter [mm/mm]')
        ax.set_ylim([0, 1.3])
    eye = 0
    for trial_data in participant_data:
        t = trial_data["data"][eye]["timestamp"]
        t = t - t[0]
        d = trial_data["data"][eye]["diameter_3d"]
        # Normalizing by the mean of the first second of data
        d = d / np.mean(d[np.where(t<1)])
        plot_topics[trial_data["attributes"]["task"]].plot(t, d, "k-")
        fig.canvas.draw()
    plt.show()