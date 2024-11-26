from argparse import ArgumentParser
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pupiltools.data_analysis as da
import pupiltools.data_plotting as d_plt

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the HDF file containing processed data."
    )
    return parser.parse_args()

def main(data_filepath: Path):
    participant_id = "P23"
    dt = 0.01
    tasks = ("action", "observation")
    variables = ("timestamp", "diameter_3d")
    diameter = dict.fromkeys(tasks)
    t = dict.fromkeys(tasks)
    trendlines = dict.fromkeys(tasks)
    with h5py.File(data_filepath, mode='r') as f_root:
        for task in tasks:
            dataset = f_root["/" + "/".join((participant_id, task))]
            data = dataset.fields(variables)
            diameter[task] = np.concatenate((data["diameter_3d"][:,:,0], data["diameter_3d"][:,:,1]), axis=1)
            trendlines[task] = da.get_trendlines_by_task(diameter[task])
            t[task] = np.arange(start=0, stop=dt*diameter[task].shape[0], step=dt)
    fig = plt.figure()
    ax = fig.subplots()
    for task in tasks:
        d_plt.plot_trendline_range(ax, t[task], trendlines[task], xlabel="Time [s]", ylabel="Fractional Change in Pupil Diameter [mm/mm]", label=task)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))