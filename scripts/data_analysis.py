import pupiltools.data_import as di

from argparse import ArgumentParser
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
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
    participant_data = {"action": [[], []], "observation": [[], []]}
    with di.GazeDataFile(file_path, mode='r') as datafile:
        for i_trial in range(datafile.n_trials):
            for eye in eyes:
                data = datafile.get_data(trial=i_trial, eye=eye, variables=variables, **hdf_path_info)
                attr = datafile.get_attributes(trial=i_trial, **hdf_path_info)
                participant_data[attr['task']][eye].append(data)
    fig, axs = plt.subplots(2,1)
    axs[0].set_title("Action")
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Diameter [mm]')
    axs[0].set_ylim([-0.3, 0.3])
    axs[1].set_title("Observation")
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Diameter [mm]')
    axs[1].set_ylim([-0.3, 0.3])
    eye = 0
    for axi, sub_data in enumerate(participant_data.values()):
        for trial in range(60):
            t = sub_data[eye][trial]["timestamp"]
            t = t - t[0]
            d = sub_data[eye][trial]["diameter_3d"]
            d = d / np.mean(d[np.where(t<1)]) - 1
            axs[axi].plot(t, d)
            fig.canvas.draw()
    plt.show()