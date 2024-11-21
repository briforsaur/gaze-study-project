from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import json
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
            print(f"Accuracy:{accuracy_score(labels, output)}")
            f1[phase][p_id] = f1_score(labels, output)
            print(f"F1      :{f1_score(labels, output)}")
    print(f"Avg Accuracy: {np.mean(acc['test'])}")
    print(f"Avg F1: {np.mean(f1['test'])}")
            



if __name__=="__main__":
    args = get_args()
    main(**vars(args))