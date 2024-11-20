from argparse import ArgumentParser
from pathlib import Path
from numpy.lib.npyio import NpzFile
import numpy as np
from sklearn.neural_network import MLPClassifier

from pupiltools.utilities import make_digit_str


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the NPZ file containing feature data."
    )
    parser.add_argument(
        "results_path", type=Path, help="Path to directory to save the results."
    )
    return parser.parse_args()


def main(data_filepath: Path, results_path: Path):
    participants = [f"P{make_digit_str(i, width=2)}" for i in range(1, 31)]
    # Load feature data file
    class_data_file = np.load(data_filepath)
    # For all participants
    for p_id in participants:
        # Split data into training and validation (leave one out)
        training_ids = [p for p in participants if p is not p_id]
        # Split data into testing and training features and class labels
        rng = np.random.default_rng()
        train_features, train_labels = get_class_data(class_data_file, training_ids, rng)
        test_features, test_labels = get_class_data(class_data_file, (p_id,))
        # Train model
        clf = MLPClassifier(max_iter=1000, learning_rate="adaptive")
        clf.fit(train_features, train_labels)
        # Get final training metrics
        train_output = clf.predict(train_features)
        train_stats = calc_performance_stats(train_output, train_labels)
        print(f"\n{p_id} Results")
        print(f"Training:\nAccuracy: {train_stats[0]:.3f}   TPR: {train_stats[1]:.3f}   TNR: {train_stats[2]:.3f}")
        # Test model on left out data
        test_output = clf.predict(test_features)
        # Get final test metrics
        test_stats = calc_performance_stats(test_output, test_labels)
        print(f"Testing:\nAccuracy: {test_stats[0]:.3f}   TPR: {test_stats[1]:.3f}   TNR: {test_stats[2]:.3f}")


def get_class_data(class_data_file: NpzFile, ids: list[str], rng: np.random.Generator = None) -> tuple[np.ndarray]:
    feature_data = None
    label_data = None
    for id in ids:
        participant_data = class_data_file[id]
        if rng is not None:
            rng.shuffle(participant_data, axis=0)
        features = participant_data[:, :-1]
        labels = participant_data[:, -1].astype(np.int64)
        if feature_data is None:
            feature_data = features
            label_data = labels
        else:
            feature_data = np.concat((feature_data, features), axis=0)
            label_data = np.concat((label_data, labels), axis=0)
    return feature_data, label_data


def calc_performance_stats(outputs: np.ndarray, labels: np.ndarray) -> tuple[float]:
    positive_pop = np.sum(labels)
    negative_pop = labels.size - positive_pop
    accuracy = np.sum(outputs == labels)/labels.size
    true_positives = np.sum(outputs*labels)
    true_positive_rate = true_positives/positive_pop
    true_negatives = np.sum(np.logical_not(outputs)*np.logical_not(labels))
    true_negative_rate = true_negatives/negative_pop
    return accuracy, true_positive_rate, true_negative_rate



if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
