from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig
import numpy as np
from .aliases import RawParticipantDataType


def plot_raw_pupil_diameter_comparison(participant_data: RawParticipantDataType) -> mpl_fig:
    fig, axs = plt.subplots(2,1, figsize=(10,10))
    fig.suptitle("Comparison of Relative Pupil Diameter Change from Baseline")
    tasks = ("action", "observation")
    plot_topics: dict[str, mpl_axes] = dict(zip(tasks, axs))
    for title, ax in plot_topics.items():
        ax.set_title(title.capitalize())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Normalized Diameter [mm/mm]')
        ax.set_ylim([-0.3, 0.3])
    eye = 0
    for trial_data in participant_data:
        t = trial_data["data"][eye]["timestamp"]
        t = t - t[0]
        d = trial_data["data"][eye]["diameter_3d"]
        # Normalizing by the mean of the first second of data
        d = d / np.mean(d[np.where(t<1)]) - 1
        plot_topics[trial_data["attributes"]["task"]].plot(t, d)
    return fig