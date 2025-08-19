# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pupiltools.constants import TASK_TYPES, PARTICIPANTS
from pupiltools.data_import import get_class_data

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the NPZ file containing feature data."
    )
    parser.add_argument(
        "--fig_path", type=Path, default=None, help="Path to save the output figures."
    )
    return parser.parse_args()


def main(data_filepath: Path, fig_path: Path):
    feature_labels = ("Mean Pupil Diameter", "Max Pupil Diameter", "Mean Absolute Pupil Rate", "Max Pupil Rate")
    feature_units = ("mm/mm", "mm/mm", "1/s", "1/s")
    class_data_file = np.load(data_filepath)
    features, labels = get_class_data(class_data_file, PARTICIPANTS)
    N_features = features.shape[1]
    labels: np.ndarray = labels == 1
    features = {
        "action": features[~labels, :],
        "observation": features[labels, :]
    }
    for i in range(N_features):
        w = 6.7
        fig = plt.figure(figsize=(w, w*9/16), dpi=300/(w/3.5))
        ax = fig.subplots()
        lims = np.zeros((2,2))
        for i_task, task in enumerate(TASK_TYPES):
            ax.hist(
                features[task][:,i],
                bins="auto",
                histtype="barstacked",
                label=task,
                alpha=0.7,
                rwidth=1
            )
            lims[i_task, :] = np.percentile(features[task][:,i], (1, 99))
        lims = np.array((lims[:,0].min(), lims[:,1].max()))
        ax.set_xlim(lims.round(1))
        ax.set_ylabel("Number of Samples")
        n_eye = i % 2
        n_feature = i // 2
        ax.set_xlabel(f"{feature_labels[n_feature]} of Eye {n_eye} [{feature_units[n_feature]}]")
        ax.legend()
        if fig_path is not None:
            if not fig_path.exists():
                fig_path.mkdir()
            fig.savefig(fig_path / f"feature_{i}.pdf", bbox_inches="tight")
    plt.show()
    


if __name__=="__main__":
    args = get_args()
    main(**vars(args))