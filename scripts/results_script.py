from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score

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
    acc = {"train": np.zeros((30,)), "test": np.zeros((30,))}
    f1 = {"train": np.zeros((30,)), "test": np.zeros((30,))}
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
    print(f"Avg Accuracy   : {np.mean(acc['test']):.3f}")
    print(f"Median Accuracy: {np.median(acc['test']):.3f}")
    fig_acc = plot_hist(acc["test"], xlabel="Accuracy", ylabel="Number of Model Instances")
    fig_acc.savefig(results_path / "accuracy_histogram.pdf")
    print(f"Avg F1         : {np.mean(f1['test']):.3f}")
    print(f"Median F1      : {np.median(f1['test']):.3f}")
    fig_f1 = plot_hist(f1["test"], xlabel="F1 Score", ylabel="Number of Model Instances")
    fig_f1.savefig(results_path / "f1_histogram.pdf")
    plt.show()
            

def plot_hist(data: np.ndarray, xlabel: str, ylabel:str) -> Figure:
    fig, ax = plt.subplots()
    ax.hist(data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig



if __name__=="__main__":
    args = get_args()
    main(**vars(args))