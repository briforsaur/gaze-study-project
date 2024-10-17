from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig, SubFigure as mpl_subfig
from numpy import typing as npt
import numpy as np
from .aliases import RawParticipantDataType, pupil_datatype


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


def resample_comparison(old_data: np.ndarray, rs_data: np.ndarray, variable: str, ylabel: str, title: str = "") -> mpl_fig:
    if not title:
        dt = rs_data["timestamp"][1]
        title = f"Comparison of Data After Resampling at {1/dt:0.1f} Hz"
    variables = (variable, "confidence")
    ylabels = (ylabel, "Confidence")
    labels = ("Raw", "Resampled")
    t_series = (old_data["timestamp"] - old_data["timestamp"][0], rs_data["timestamp"])
    d_series = (old_data, rs_data)
    fig = plt.figure(figsize=(10,8))
    axs = fig.subplots(2, 1)
    fig.suptitle(title)
    ax: mpl_axes
    for ax, var, y_lbl in zip(axs, variables, ylabels):
        for t, d, lbl in zip(t_series, d_series, labels):
            ax.plot(t, d[var], label=lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_lbl)
        ax.legend()
    ax.set_ylim(0.6, 1.1)
    return fig


def plot_dt_histogram(time_deltas: list[np.ndarray]) -> mpl_fig:
    dt_array = np.concatenate(time_deltas)
    fig, ax = plt.subplots()
    fig.suptitle("Probability Mass Function of Time Between Samples")
    weights = np.ones_like(dt_array)/dt_array.size
    ax.hist(dt_array, weights=weights)
    ax.set_xlabel("Time Between Samples [s]")
    ax.set_ylabel("Proportion of Samples")
    return fig