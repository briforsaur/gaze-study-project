# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from matplotlib import pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig, SubFigure as mpl_subfig
from numpy import typing as npt
import numpy as np
from .aliases import RawParticipantDataType, pupil_datatype
from .constants import TASK_TYPES


_cm = 1 / 2.54


def plot_raw_pupil_diameter_comparison(
    participant_data: RawParticipantDataType,
) -> mpl_fig:
    """Plot normalized diameter vs time for all trials split by eye and task type

    Produces a figure with a 2x2 grid of subplots showing pupil diameter normalized to
    the first 1 second of data for each trial versus time. The left and right columns
    show the results for eyes 0 (participant's right) and 1 (participant's left), 
    respectively, while the top and bottom rows show the results for action and
    observation tasks, respectively.

    Parameters
    ----------
    participant_data: :py:type:`pupiltools.aliases.RawParticipantDataType`
        A list of trials with metadata and data.

    Returns
    -------
    matplotlib.Figure
        Handle to the created figure.
    """
    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(1, 2)
    fig.suptitle("Comparison of Relative Pupil Diameter Change from Baseline")
    subfig: mpl_subfig
    for eye, subfig in enumerate(subfigs):
        axs = subfig.subplots(2, 1)
        plot_topics: dict[str, mpl_axes] = dict(zip(TASK_TYPES, axs))
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
    """Plot a comparison of one data variable and confidence before and after resampling

    Produces a figure with 2x1 subplots. The top subplot shows the selected variable
    over time before and after resampling. The bottom subplot shows the pupil detection
    confidence over time before and after resampling.

    Parameters
    ----------
    old_data: np.ndarray
        A numpy record array of variables. Must include the variables ``'timestamp'``, 
        ``'confidence'``, and the value of the parameter ``variable``.
    rs_data: np.ndarray
        A numpy record array of variables. Must include the same three values required
        by ``old_data``.
    variable: str
        The name of the variable to plot in the top subplot. For example, 
        ``"diameter_3d"``.
    ylabel: str
        The label for the y-axis of the top subplot.
    title: str, default=""
        The title of the plot.
    """
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
    """Plot a histogram of time differences between samples

    Produces a histogram showing the distribution of time differences between samples
    across a number of trials.

    Parameters
    ----------
    time_deltas: list[np.ndarray]
        A list of arrays of time differences between each samples.
    
    Returns
    -------
    matplotlib.Figure
        Handle to the created figure.
    """
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

    Produces a figure with 2x1 subplots showing the average pupil dilation over time for
    all trials of the same task type, with shaded regions showing the distribution of
    the pupil dilation.

    See also :py:func:`plot_trendline_range`.

    Parameters
    ----------
    t: dict[str, numpy.ndarray[numpy.float64]]
        A dictionary of arrays of shape N, representing the time of each sample. The
        dictionary keys are the task type names, e.g. ``("action", "observation")``.
    trendline_array: dict[str, numpy.ndarray[numpy.float64]]
        A dictionary of Nx2x3 arrays, where the first dimension is the samples in time,
        the second dimension is the eye number (0 for right, 1 for left), and the third
        dimension stacks the different statistics (0 for bottom range, 1 for mean or
        median - what will be displayed as a line, and 1 for top range). The dictionary 
        keys are the task type names, matching the keys for ``t``.
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

    Produces a plot of a line surrounded by shaded regions. The range of the shaded 
    regions are determined by the ``Y`` parameter. If the regions are intended to 
    represent 5th and 95th percentiles, that calculation must be performed first.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes on which to plot.
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
    """Plot a histogram of values separated by task type

    Plots a figure with 2x1 subplots. The subplots show histograms of the maximum pupil
    diameter change as a fraction of the mean pupil diameter for the first 1 second,
    separated by the keys of the input data dictionary. The top and bottom subplots show
    the distribution for eye 0 (participant's right eye) and 1 (participant's left eye),
    respectively.

    Parameters
    ----------
    max_values: dict[str, np.ndarray]
        A dictionary of numpy arrays, where the keys of the dictionary are the separate
        groups to be plotted, e.g. the task types ``("action", "observation")``.
    """
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


def manual_hist(
    values: np.ndarray,
    bin_edges: np.ndarray,
    split: float | None = None,
    normalize: bool = False,
):
    """Plot a histogram of the maximum change in pupil diameter between task types

    Produces a histogram with two sets of columns, one column colour representing action
    and the other observation tasks, showing the distribution of maximum fractional
    change in pupil diameter.

    This function allows two datasets to be plotted as histograms with the same bin
    sizes and placements for easy comparison.

    Parameters
    ----------
    values: np.ndarray
        An Nx2 array of maximum pupil diameter change, where N is the number of trials
        and columns 0 and 1 are the values for action and observation trials,
        respectively.
    bin_edges: np.ndarray
        An array of floats representing the left edges of the bars in the histogram.
    split: float | None, default=None
        The value that maximally splits the two classes, represented as a dashed,
        vertical line on the plot. Default does not show a line.
    normalize: bool = False
        Flag to indicate whether the values should be normalized to show a proportion
        of trials. If false, the histogram shows the total number of trials.
    """
    fig, ax = plt.subplots(1, figsize=(16, 9))
    if normalize:
        values = values / values.sum(axis=0)
        y_label = "Proportion of Trials"
    else:
        y_label = "Number of Trials"
    for i in (0, 1):
        ax.bar(
            bin_edges[:-1],
            values[:, i],
            0.025,
            align="edge",
            alpha=0.8,
            label=TASK_TYPES[i],
        )
    if split is not None:
        ax.axvline(split, color="k", ls="--")
    plt.legend()
    ax.set_xlabel("Maximum Fractional Change in Pupil Diameter")
    ax.set_ylabel(y_label)
    fig.suptitle("Comparison of Pupil Diameter Change By Task Type")
