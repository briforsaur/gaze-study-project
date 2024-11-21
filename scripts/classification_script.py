from argparse import ArgumentParser
from pathlib import Path
from numpy.lib.npyio import NpzFile
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from pupiltools.utilities import make_digit_str, get_datetime


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
    results_path = results_path / get_datetime()
    if not results_path.exists():
        results_path.mkdir()
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
        save_model(clf, path=(results_path / f"models/{p_id}_left_out.pickle"))
        # Get final training metrics
        train_output = clf.predict(train_features)
        train_acc = accuracy_score(train_labels, train_output)
        train_f1 = f1_score(train_labels, train_output)
        print(f"\n{p_id} Results")
        print(f"Training Accuracy: {train_acc:.3f}")
        # Test model on left out data
        test_output = clf.predict(test_features)
        # Get final test metrics
        test_acc = accuracy_score(test_labels, test_output)
        test_f1 = f1_score(test_labels, test_output)
        print(f"Testing Accuracy: {test_acc:.3f}")


def get_class_data(class_data_file: NpzFile, ids: list[str], rng: np.random.Generator = None) -> tuple[np.ndarray]:
    """Extract labelled feature data from a .npz file
    
    Parameters
    ----------
    class_data_file: numpy.lib.npyio.NpzFile
        An NpzFile object returned by numpy.load containing a set of arrays of labelled 
        feature data for each participant. Each array is expected to be of shape
        (n_samples, n_features + 1), such that the last column contains the label data.
        The label data is expected to be cast-able to the integer type.
    ids: list[str]
        A list of identifiers for the arrays to be extracted from the npz file.
    rng: np.random.Generator, optional
        A numpy random number generator used to shuffle all the rows of the extracted
        data, ensuring a random sample order while keeping features grouped with the 
        corresponding labels. If this is not provided, the samples will be in order of
        the identifiers provided and the order they were in the original arrays.
    
    Returns
    -------
    feature_data: numpy.ndarray
        An array of shape (N, n_features) containing the features for all samples in
        the extracted arrays, where N is the sum of all n_samples from all arrays. The 
        samples may be randomized if an rng was provided.
    label_data: numpy.ndarray
        An array of shape (N,) containing the labels for all samples in the extracted
        arrays. Each label element i corresponds to row i of the feature_data array.
    """
    labelled_feature_data = None
    for id in ids:
        participant_data = class_data_file[id]
        if labelled_feature_data is None:
            labelled_feature_data = participant_data
        else:
            labelled_feature_data = np.concat(
                (labelled_feature_data, participant_data), axis=0
            )
    if rng is not None:
        rng.shuffle(labelled_feature_data, axis=0)
    return labelled_feature_data[:,:-1], labelled_feature_data[:,-1].astype(np.int64)


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
    with open(path, 'wb') as f:
        pickle.dump(model, file=f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
