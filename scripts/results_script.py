from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the json file containing the results data."
    )
    parser.add_argument(
        "results_path", type=Path, help="Path to directory to save the figures and other outputs."
    )
    return parser.parse_args()


def main(data_filepath, results_path):
    with open(data_filepath, "r") as f:
        classification_results: dict[str, dict] = json.load(f)
    N_c = len(classification_results)
    acc = {"train": np.zeros((N_c,)), "test": np.zeros((N_c,))}
    f1 = {"train": np.zeros((N_c,)), "test": np.zeros((N_c,))}
    confusion_matrices = {"train": np.zeros((2, 2, N_c)), "test": np.zeros((2, 2, N_c))}
    for p_id, p_results in enumerate(classification_results.values()):
        print(p_id)
        for phase, results in p_results.items():
            labels = np.array(results["labels"])
            output = np.array(results["output"])
            print(phase)
            acc[phase][p_id] = accuracy_score(labels, output)
            print(f"Accuracy:{acc[phase][p_id]:.3f}")
            f1[phase][p_id] = f1_score(labels, output)
            print(f"F1      :{f1[phase][p_id]:.3f}")
            confusion_matrices[phase][:,:,p_id] = confusion_matrix(labels, output)
    print(f"Avg Accuracy   : {np.mean(acc['test']):.3f}")
    print(f"Median Accuracy: {np.median(acc['test']):.3f}")
    fig_acc = plot_hist(acc["test"], xlabel="Accuracy", ylabel="Number of Model Instances")
    fig_acc.savefig(results_path / "accuracy_histogram.pdf")
    print(f"Avg F1         : {np.mean(f1['test']):.3f}")
    print(f"Median F1      : {np.median(f1['test']):.3f}")
    fig_f1 = plot_hist(f1["test"], xlabel="F1 Score", ylabel="Number of Model Instances")
    fig_f1.savefig(results_path / "f1_histogram.pdf")
    fig_conf_train = plot_confusion(np.mean(confusion_matrices["train"], axis=2), axlabels=("action", "observation"), clabel="Number of Samples")
    fig_conf_train.savefig(results_path / "confusion_matrix_training.pdf")
    fig_conf_test = plot_confusion(np.mean(confusion_matrices["test"], axis=2), axlabels=("action", "observation"), clabel="Number of Samples")
    fig_conf_test.savefig(results_path / "confusion_matrix_testing.pdf")
    plt.show()
            

def plot_hist(data: np.ndarray, xlabel: str, ylabel:str) -> Figure:
    w = 6.7
    fig = plt.figure(figsize=(w, w*9/16), dpi=300/(w/3.5))
    ax = fig.subplots()
    ax.hist(data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_confusion(data: np.ndarray, axlabels: tuple[str], clabel: str) -> Figure:
    fig, ax = plt.subplots()
    n = len(axlabels)
    im = ax.imshow(data)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(clabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(n), labels=axlabels)
    ax.set_xlabel("Predicted Label")
    ax.set_yticks(np.arange(n), labels=axlabels, rotation=90, va="center")
    ax.set_ylabel("True Label")
    # Create text annotations on each cell
    for i in range(n):
        for j in range(n):
            ax.text(j,i, f"{data[i, j]:.1f}\n({100*data[i, j]/np.sum(data):.1f}%)", ha="center", va="center", color="k", backgroundcolor="w")
    return fig


if __name__=="__main__":
    args = get_args()
    main(**vars(args))