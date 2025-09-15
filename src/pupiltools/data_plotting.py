# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig, SubFigure as mpl_subfig
from numpy import typing as npt
import numpy as np
from .aliases import RawParticipantDataType, pupil_datatype


_cm = 1 / 2.54


def plot_raw_pupil_diameter_comparison(
    participant_data: RawParticipantDataType,
) -> mpl_fig:
    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(1, 2)
    fig.suptitle("Comparison of Relative Pupil Diameter Change from Baseline")
    tasks = ("action", "observation")
    subfig: mpl_subfig
    for eye, subfig in enumerate(subfigs):
        axs = subfig.subplots(2, 1)
        plot_topics: dict[str, mpl_axes] = dict(zip(tasks, axs))
        for title, ax in plot_topics.items():
            ax.set_title(title.capitalize())
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Normalized Diameter [mm/mm]")
            ax.set_ylim((-0.3, 0.3))
            ax.grid()
        for trial_data in participant_data:
            t = trial_data["data"][eye]["timestamp"] #type:ignore
            assert isinstance(t, np.ndarray)
            t = t - t[0]
            d = trial_data["data"][eye]["diameter_3d"] #type:ignore
            assert isinstance(d, np.ndarray)
            # Normalizing by the mean of the first second of data
            d = d / np.mean(d[np.where(t < 1)]) - 1
            task: str = trial_data["attributes"]["task"] #type:ignore
            plot_topics[task].plot(t, d)
    return fig


def resample_comparison(
    old_data: np.ndarray,
    rs_data: np.ndarray,
    variable: str,
    ylabel: str,
    title: str = "",
) -> mpl_fig:
    if not title:
        dt = rs_data["timestamp"][1]
        title = f"Comparison of Data After Resampling at {1/dt:0.1f} Hz"
    variables = (variable, "confidence")
    ylabels = (ylabel, "Confidence")
    labels = ("Raw", "Resampled")
    t_series = (old_data["timestamp"] - old_data["timestamp"][0], rs_data["timestamp"])
    d_series = (old_data, rs_data)
    fig = plt.figure(figsize=(10, 8))
    axs = fig.subplots(2, 1)
    fig.suptitle(title)
    ax: mpl_axes
    for ax, var, y_lbl in zip(axs, variables, ylabels):
        for t, d, lbl in zip(t_series, d_series, labels):
            ax.plot(t, d[var], label=lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_lbl)
        ax.legend()
    ax.set_ylim(0.6, 1.1) #type:ignore
    return fig


def plot_dt_histogram(time_deltas: list[np.ndarray]) -> mpl_fig:
    dt_array = np.concatenate(time_deltas)
    fig, ax = plt.subplots()
    fig.suptitle("Probability Mass Function of Time Between Samples")
    weights = np.ones_like(dt_array) / dt_array.size
    ax.hist(dt_array, weights=weights)
    ax.set_xlabel("Time Between Samples [s]")
    ax.set_ylabel("Proportion of Samples")
    return fig


def plot_trendlines(
    t: dict[str, npt.NDArray[np.float64]],
    trendlines: dict[str, npt.NDArray[np.float64]],
    title: str = "",
):
    """Plot a comparison of pupil diameter for actions and observations for both eyes

    Parameters
    ----------
    t: dict[str, numpy.ndarray[numpy.float64]]
        A dictionary of arrays of shape N, representing the time of each sample.
    trendline_array: dict[str, numpy.ndarray[numpy.float64]]
        A dictionary of Nx2x3 arrays, where the first dimension is the samples in time,
        the second dimension is the eye number (0 for right, 1 for left), and the third
        dimension stacks the different statistics (0 for bottom range, 1 for mean or
        median - what will be displayed as a line, and 1 for top range).
    """
    fig, axs = plt.subplots(2, figsize=(30 * _cm, 25 * _cm))
    fig.suptitle(title)
    ax: mpl_axes
    for i, ax in enumerate(axs):
        for task, trendline_array in trendlines.items():
            plot_trendline_range(
                ax,
                t[task],
                trendline_array[:, i, :],
                xlabel="Time [s]",
                ylabel="Fractional Change in Pupil Diameter",
                xlim=(0., 4.25),
                title=f"eye{i}",
                label=task,
                alpha=0.2,
            )


def plot_trendline_range(
    ax: mpl_axes,
    x: np.ndarray,
    Y: np.ndarray,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    title: str = "",
    label: str = "",
    alpha: float = 0.2,
):
    """Plot a trendline with percentiles as shaded regions

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis on which to plot.
    x: np.ndarray
        A 1-D array of length N representing the x-axis values.
    Y: np.ndarray
        An N x 3 array representing the trendline and shaded region bounds. The first
        column is the lower bounds of the shaded region, the second is the trendline,
        and the third is the upper bounds of the shaded region.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    xlim: tuple[float] = None
        The x-axis limits, as in the matplotlib ax.set_xlim method.
    title: str = ""
        The axes title. Blank by default.
    label: str = ""
        Label to use for the trendline in the legend.
    alpha: float = 0.2
        The alpha (transparency) of the shaded region.
    """
    ax.set_title(title)
    ax.plot(x, Y[:, 1], label=label)
    ax.fill_between(x, Y[:, 0], Y[:, 2], alpha=alpha)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(visible=True, which="major")
    vline_style = {"alpha": 0.8, "color": "k", "linestyle": "--"}
    ax.axvline(x=1.0, **vline_style)
    ax.axvline(x=2.5, **vline_style)
    ax.text(x=1.0, y=Y.max()*0.8, s="Instruction Start", horizontalalignment="right", verticalalignment="top", rotation="vertical")
    ax.text(x=2.5, y=Y.max()*0.8, s="Instruction End", horizontalalignment="right", verticalalignment="top", rotation="vertical")


def plot_max_values(max_values: dict[str, np.ndarray]):
    fig1, axs = plt.subplots(2, figsize=(16, 9))
    fig1.suptitle("Distribution of Maximum Fractional Pupil Diameter Change")
    ax: mpl_axes
    for i, ax in enumerate(axs):
        ax.set_title(f"eye{i}")
        for task, max_val_array in max_values.items():
            # weights = np.ones_like(max_values[:,i,j])/max_values[:,i,j].size
            ax.hist(
                max_val_array[:, i],
                bins="auto",
                histtype="barstacked",
                label=task,
                alpha=0.7,
            )
        ax.legend()
        ax.set_xlabel("Max Fractional Change in Pupil Diameter")
        ax.set_ylabel("Proportion of Trials")
    # TODO: FIX BROKEN CODE BELOW
    # fig2, ax = plt.subplots(1, figsize=(16, 9))
    # for j in (0, 1):
    #     max_values_sum = max_values[:,:,j].sum(axis=1)
    #     #weights = np.ones_like(max_values_sum)/max_values_sum.size
    #     ax.hist(max_values_sum, bins="auto", histtype="barstacked", label=task_labels[j], alpha=0.7)


def manual_hist(
    values: np.ndarray,
    bin_edges: np.ndarray,
    split: float | None = None,
    normalize: bool = False,
):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    task_labels = ["action", "observation"]
    if normalize:
        values = values / values.sum(axis=0)
    for i in (0, 1):
        ax.bar(
            bin_edges[:-1],
            values[:, i],
            0.025,
            align="edge",
            alpha=0.8,
            label=task_labels[i],
        )
    if split is not None:
        ax.axvline(split, color="k", ls="--")
    plt.legend()
    ax.set_xlabel("Maximum Fractional Change in Pupil Diameter")
    ax.set_ylabel("Proportion of Trials")
    fig.suptitle("Comparison of Pupil Diameter Change By Task Type")
