from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from time import time

from pupiltools.utilities import get_datetime
from pupiltools.data_import import get_class_data
from pupiltools.constants import PARTICIPANTS


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
    # Load feature data file
    class_data_file = np.load(data_filepath)
    features, class_labels, group_labels = get_class_data(class_data_file, PARTICIPANTS)
    # Initialize model
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, learning_rate="adaptive"
    )
    # Define leave-one-out cross validation splits
    cv_scheme = LeaveOneGroupOut().split(features, class_labels, group_labels)
    # Perform cross validation
    scores = cross_validate(
        clf,
        features,
        class_labels,
        cv=cv_scheme,
        scoring=("accuracy", "f1"),
        return_train_score=True,
        return_estimator=True,
        verbose=1,
    )
    estimators = scores["estimator"]
    scores = {key: value for key, value in scores.items() if key != "estimator"}
    print(
        f"Mean train accuracy: {scores['train_accuracy'].mean()}\nStd. Dev.: {scores['train_accuracy'].std()}"
    )
    print(
        f"Mean test accuracy: {scores['test_accuracy'].mean()}\nStd. Dev.: {scores['test_accuracy'].std()}"
    )
    print(
        f"Mean train f1: {scores['train_f1'].mean()}\nStd. Dev.: {scores['train_f1'].std()}"
    )
    print(
        f"Mean test f1: {scores['test_f1'].mean()}\nStd. Dev.: {scores['test_f1'].std()}"
    )
    # Get final training metrics
    classification_results = {
        "date_time": datetime.now().isoformat(),
        "input_dimensions": features.shape[1:],
        "classifier_parameters": clf.get_params(),
        "cross_validation_scores": scores,
        "training_loss": [estimator.loss_curve_ for estimator in estimators],
    }
    results_path = results_path / get_datetime()
    if not results_path.exists():
        results_path.mkdir(parents=True)
    save_as_json(classification_results, path=(results_path / "cv_results.json"))


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
        json.dump(results, f, indent=4, default=np_default)


def np_default(o):
    try:
        o_list = o.tolist()
    except:
        raise (TypeError)
    return o_list


if __name__ == "__main__":
    args = get_args()
    start = time()
    main(**vars(args))
    end = time()
    print(round(end - start))
