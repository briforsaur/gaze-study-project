from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig, SubFigure as mpl_subfig
import numpy as np
from .aliases import RawParticipantDataType


def plot_raw_pupil_diameter_comparison(participant_data: RawParticipantDataType) -> mpl_fig:
    fig = plt.figure(figsize=(15,10))
    subfigs = fig.subfigures(1,2)
    fig.suptitle("Comparison of Relative Pupil Diameter Change from Baseline")
    tasks = ("action", "observation")
    subfig: mpl_subfig
    for eye, subfig in enumerate(subfigs):
        axs = subfig.subplots(2,1)
        plot_topics: dict[str, mpl_axes] = dict(zip(tasks, axs))
        for title, ax in plot_topics.items():
            ax.set_title(title.capitalize())
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Normalized Diameter [mm/mm]')
            ax.set_ylim([-0.3, 0.3])
            ax.grid()
        for trial_data in participant_data:
            t = trial_data["data"][eye]["timestamp"]
            t = t - t[0]
            d = trial_data["data"][eye]["diameter_3d"]
            # Normalizing by the mean of the first second of data
            d = d / np.mean(d[np.where(t<1)]) - 1
            task: str = trial_data["attributes"]["task"]
            plot_topics[task].plot(t, d)
    return fig