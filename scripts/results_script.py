# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pupiltools.utilities import make_digit_str

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

W = 6.7


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath",
        type=Path,
        help="Path to the json file containing the results data.",
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to directory to save the figures and other outputs.",
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Flag to show plots at end of script."
    )
    return parser.parse_args()


def main(data_filepath: Path, results_path: Path, show_plots: bool):
    with open(data_filepath, "r") as f:
        classification_results: dict[str, dict] = json.load(f)
    cv_scores = classification_results["cross_validation_scores"]
    loss_curves = classification_results["training_loss"]
    N_c = len(cv_scores["test_accuracy"])
    acc = {
        "train": np.array(cv_scores["train_accuracy"]),
        "test": cv_scores["test_accuracy"],
    }
    f1 = {"train": np.array(cv_scores["train_f1"]), "test": cv_scores["test_f1"]}
    if not results_path.exists():
        results_path.mkdir(parents=True)
    print(f"Avg Accuracy   : {np.mean(acc['test']):.3f}")
    print(f"Median Accuracy: {np.median(acc['test']):.3f}")
    fig_acc = plot_hist(
        acc["test"], xlabel="Accuracy", ylabel="Number of Model Instances"
    )
    fig_acc.savefig(results_path / "accuracy_histogram.pdf", bbox_inches="tight")
    print(f"Avg F1         : {np.mean(f1['test']):.3f}")
    print(f"Median F1      : {np.median(f1['test']):.3f}")
    fig_f1 = plot_hist(
        f1["test"], xlabel="F1 Score", ylabel="Number of Model Instances"
    )
    fig_f1.savefig(results_path / "f1_histogram.pdf", bbox_inches="tight")
    fig_loss_all = plt.figure(figsize=(W, W * 9 / 16), dpi=300 / (W / 3.5))
    ax_all = fig_loss_all.subplots()
    for participant_id, loss_curve in enumerate(loss_curves):
        p_label = make_digit_str(participant_id, width=2)
        title = f"Training Loss for Model with P{p_label} Left Out"
        fig_loss = plt.figure(figsize=(W, W * 9 / 16), dpi=300 / (W / 3.5))
        plot_loss(fig_loss, loss_curve, title)
        fig_loss.savefig(results_path / f"loss_curve_P{p_label}_left_out.pdf")
        if not show_plots:
            plt.close(fig_loss)
        ax_all.plot(loss_curve)
    ax_all.set_xlabel("Epochs")
    ax_all.set_ylabel("Training (Logistic) Loss")
    ax_all.set_title(f"Training Loss for All Models")
    fig_loss_all.savefig(results_path / f"loss_curves_all.pdf")
    if show_plots:
        plt.show()


def plot_hist(data: np.ndarray, xlabel: str, ylabel: str) -> Figure:
    fig = plt.figure(figsize=(W, W * 9 / 16), dpi=300 / (W / 3.5))
    ax = fig.subplots()
    ax.hist(data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_confusion(data: np.ndarray, axlabels: tuple[str], clabel: str) -> Figure:
    fig, ax = plt.subplots()
    n = len(axlabels)
    im = ax.imshow(data)
    cbar = ax.figure.colorbar(im, ax=ax)  # type:ignore
    cbar.ax.set_ylabel(clabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(n), labels=axlabels)
    ax.set_xlabel("Predicted Label")
    ax.set_yticks(np.arange(n), labels=axlabels, rotation=90, va="center")
    ax.set_ylabel("True Label")
    # Create text annotations on each cell
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}\n({100*data[i, j]/np.sum(data):.1f}%)",
                ha="center",
                va="center",
                color="k",
                backgroundcolor="w",
            )
    return fig


def plot_loss(fig: Figure, data, title, ax = None):
    if ax is None:
        ax = fig.subplots()
    ax.plot(data)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training (Logistic) Loss")
    ax.set_title(title)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
