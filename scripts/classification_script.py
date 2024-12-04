from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pickle
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from pupiltools.utilities import make_digit_str, get_datetime
from pupiltools.data_import import get_class_data


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the NPZ file containing feature data."
    )
    parser.add_argument(
        "results_path", type=Path, help="Path to directory to save the results."
    )
    parser.add_argument(
        "hidden_layer_sizes", type=int, nargs="*", help="List of hidden layer sizes"
    )
    return parser.parse_args()


def main(data_filepath: Path, results_path: Path, hidden_layer_sizes: list[int]):
    results_path = results_path / get_datetime()
    if not results_path.exists():
        results_path.mkdir()
    participants = [f"P{make_digit_str(i, width=2)}" for i in range(1, 31)]
    # Load feature data file
    class_data_file = np.load(data_filepath)
    classification_results = {}
    # For all participants
    for p_id in participants:
        # Split data into training and validation (leave one out)
        training_ids = [p for p in participants if p is not p_id]
        # Split data into testing and training features and class labels
        rng = np.random.default_rng()
        train_features, train_labels = get_class_data(
            class_data_file, training_ids, rng
        )
        test_features, test_labels = get_class_data(class_data_file, (p_id,))
        # Train model
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, learning_rate="adaptive")
        clf.fit(train_features, train_labels)
        save_model(clf, path=(results_path / f"models/{p_id}_left_out.pickle"))
        # Get final training metrics
        train_output = clf.predict(train_features)
        train_acc = accuracy_score(train_labels, train_output)
        train_f1 = f1_score(train_labels, train_output)
        print(f"\n{p_id} Results")
        print(f"Training Accuracy: {train_acc:.3f}   Training F1: {train_f1:.3f}")
        # Test model on left out data
        test_output = clf.predict(test_features)
        # Get final test metrics
        test_acc = accuracy_score(test_labels, test_output)
        test_f1 = f1_score(test_labels, test_output)
        print(f"Testing Accuracy: {test_acc:.3f}   Testing F1: {test_f1:.3f}")
        classification_results.update(
            {
                p_id: {
                    "train": {"labels": train_labels.tolist(), "output": train_output.tolist()},
                    "test": {"labels": test_labels.tolist(), "output": test_output.tolist()},
                }
            }
        )
    save_as_json(classification_results, path=(results_path / "results.json"))


def save_model(model, path: Path):
    """Pickle a Python object as bytes

    Parameters
    ----------
    model: Any
        Any python object, but intended to be a neural network or other model generated
        by scikit-learn.
    path: pathlib.Path
        A full path to the desired file to write.
    """
    if not path.parent.exists():
        path.parent.mkdir()
    with open(path, "wb") as f:
        pickle.dump(model, file=f, protocol=pickle.HIGHEST_PROTOCOL)

def save_as_json(results, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
