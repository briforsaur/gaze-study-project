import argparse
import h5py
from numpy.lib.npyio import NpzFile
import numpy as np
import os
from pathlib import Path
from typing import TypeAlias
from .aliases import pupil_datatype, AttributesType, RawParticipantDataType, ResampledParticipantDataType
from .utilities import make_digit_str


class GazeDataFile:
    """Object for handling gaze study HDF5 data files

    This class serves as a user interface for getting data and attributes from an HDF5
    file with a specific structure.

    Expects an HDF5 file in the following structure:

            root/
            |-- trials/
            |   |-- 000/
            |   |   |-- pupil_eye0_3d
            |   |   |-- pupil_eye1_3d
            |   |-- 001/
            |   |   |-- pupil_eye0_3d
            |   |   |-- pupil_eye1_3d
    """

    def __init__(self, file: str | bytes | os.PathLike, mode: str = 'r'):
        """Instantiate an instance of a GazeDataFile object
        
        Parameters
        ----------
        file: str | bytes | pathlike
            Path or file-like object to the hdf5 file
        mode: str
            File opening mode. Only supports modes supported by h5py's File object:

            r   Readonly, file must exist (default)
            r+  Read/write, file must exist
            w   Create file, truncate if exists
            w-  or x Create file, fail if exists
            a   Read/write if exists, create otherwise
        """
        self.file = file
        self.hdf_root = h5py.File(file, mode=mode)
        self.n_trials = len(self.hdf_root["trials"].keys())

    def get_data(
        self,
        group: str = "trials",
        trial: int = 0,
        topic: str = "pupil",
        eye: int = -1,
        method: str = "3d",
        variables: str | list[str] = "all",
    ) -> np.ndarray:
        """Get a numpy array from a gaze study dataset.

        Extract a variable or set of variables from an HDF5 file containing gaze study 
        datasets. 

        Parameters
        ----------
        group: str, default="trials"
            The data group, can be either "trials" or [TODO] "calibrations".
        trial: int, default=0
            The trial or calibration number.
        topic: str, default="pupil"
            The data topic, can be either "pupil" or [TODO] "gaze".
        eye: {0, 1}, default=0
            The eye number. 0 for participant's right, 1 for left.
        method: str, default="3d"
            The method used to track the pupil. Can be either "3d" or [TODO] "2d".
        variables: str | list[str], default="all"
            The variables to extract from the dataset. Can either be a single string 
            representing a single variable's name, or a list of strings representing the 
            names of multiple variables. The variable names can be any top-level 
            variable name in the dataset, for example "timestamp" or "norm_pos".
        
        Returns
        -------
        numpy.ndarray
            A structured array containing the requested variables within the dataset. 
        """
        datapath = get_path(group=group, trial=trial, topic=topic, eye=eye, method=method)
        dataset: h5py.Dataset = self.hdf_root[datapath]
        if isinstance(variables, str) and variables == "all":
            # Preallocate an array and read the dataset in directly to avoid an
            # intermediate copy of the entire dataset
            data = np.empty(shape=dataset.shape, dtype=pupil_datatype)
            dataset.read_direct(data)
        else:
            data = dataset.fields(variables)[:]
        return data
    
    def get_attributes(
        self, 
        group: str = "",
        trial: int = -1,
        topic: str = "",
        eye: int = -1,
        method: str = ""
    ) -> AttributesType:
        """Get the attributes of a member of the HDF File"""
        path = get_path(group=group, trial=trial, topic=topic, eye=eye, method=method)
        # Need to use dict() to avoid shallow copy that is left dangling on file close
        return dict(self.hdf_root[path].attrs)
    
    def close(self):
        self.hdf_root.close()

    def __enter__(self):
        # Used for resource management with the "with" statement
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Make sure all cleanup is done when exiting the "with" statement
        self.close()


def get_path(
    group: str = "",
    trial: int = -1,
    topic: str = "",
    eye: int = -1,
    method: str = ""
) -> str:
    """Generate an internal HDF path to the desired group or dataset
    
    All parameters are empty or invalid by default. If left as defaults, they will
    not be included in the path.

    Examples:

    >>> get_path()
    '/'
    >>> get_path("/")
    '/'
    >>> get_path("root")
    '/'
    >>> get_path(group="trials")
    '/trials'
    >>> get_path(group="trials", trial=0)
    '/trials/000'
    >>> get_path(group="trials", trial=0, topic="pupil", eye=0, method="3d")
    '/trials/000/pupil_eye0_3d'
    >>> get_path(group="trials", trial=0, topic="pupil", method="3d")

    """
    trial_str = make_digit_str(trial, width=3) if trial >= 0 else ""
    dataset_name = get_dataset_name(topic, eye, method)
    # Creating a list of non-empty path members
    pathlist = [var for var in [group, trial_str, dataset_name] if var]
    if group in ["root", "/"]:
        path = "/"
    else:
        path = "/" + "/".join(pathlist)
    return path


def get_dataset_name(topic: str = "", eye: int = -1, method: str = "") -> str:
    eye_str = f"eye{eye}" if eye in [0, 1] else ""
    # Making the dataset name out of nonempty string parts
    dataset_name_list = [part for part in [topic, eye_str, method] if part]
    return "_".join(dataset_name_list)


def get_raw_participant_data(file: str | bytes | os.PathLike, group: str = "trials", topic: str = "pupil", method: str = "3d", variables: str | list[str] = "all") -> tuple[RawParticipantDataType, dict]:
    """Get raw participant data from an HDF File"""
    hdf_path_info = {"group": group, "topic": topic, "method": method}
    participant_data = []
    with GazeDataFile(file, mode='r') as datafile:
        participant_metadata = datafile.get_attributes()
        for i_trial in range(datafile.n_trials):
            attr = datafile.get_attributes(group=group, trial=i_trial)
            data = get_eye_data(datafile, variables=variables, trial=i_trial, **hdf_path_info)
            participant_data.append({"attributes": attr, "data": data})
    return participant_data, participant_metadata


def get_resampled_participant_data(file: str | bytes | os.PathLike, group: str = "trials", topic: str = "pupil", method: str = "3d", variables: str | list[str] = "all") -> tuple[ResampledParticipantDataType, dict]:
    """Get resampled participant data from an HDF File"""
    # Just a wrapper for get_raw_participant data with the correct return type hint
    hdf_path_info = {"group": group, "topic": topic, "method": method}
    return get_raw_participant_data(file=file, variables=variables, **hdf_path_info)


def get_eye_data(datafile: GazeDataFile, variables: str | list[str], group: str = "trials", trial: int = 0, topic: str = "pupil", method: str = "3d", eyes: tuple[int] = (0, 1)) -> np.ndarray | list[np.ndarray]:
    group_path = get_path(group=group, trial=trial)
    group_obj: h5py.Group = datafile.hdf_root[group_path]
    data = []
    for eye in eyes:
        if get_dataset_name(topic=topic, eye=eye, method=method) in group_obj:
            data.append(datafile.get_data(group=group, trial=trial, topic=topic, eye=eye, method=method, variables=variables))
    # If the data list is still empty, then the eyes must be in a single dataset
    if not data:
        data = datafile.get_data(group=group, trial=trial, topic=topic, eye=-1, method=method, variables=variables)
    return data


def get_class_data(class_data_file: NpzFile, ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    feature_data: numpy.ndarray
        An array of shape (N, n_features) containing the features for all samples in
        the extracted arrays, where N is the sum of all n_samples from all arrays.
    label_data: numpy.ndarray
        An array of shape (N,) containing the class labels for all samples in the 
        extracted arrays. Each label element i corresponds to row i of the feature_data 
        array.
    group_labels: np.ndarray
        An array of shape (N,) containing the group labels for all samples.
    """
    labelled_feature_data = None
    splits = []
    for id in ids:
        participant_data = class_data_file[id]
        n_id = int(id[1:])
        group_label = np.full((participant_data.shape[0],), n_id)
        if labelled_feature_data is None:
            labelled_feature_data = participant_data
            group_labels = group_label
        else:
            labelled_feature_data = np.concat(
                (labelled_feature_data, participant_data), axis=0
            )
            group_labels = np.concat((group_labels, group_label))
    return labelled_feature_data[:, :-1], labelled_feature_data[:, -1].astype(np.int64), group_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file_path", type=Path, help="Path to the HDF5 data file")
    args = parser.parse_args()
    test_set = {"topic": "pupil", "eye": 0, "method": "3d"}
    with GazeDataFile(args.data_file_path, mode="r") as datafile:
        data = datafile.get_data(
            variables=["timestamp", "world_index", "diameter_3d"], **test_set
        )
        root_attrs = datafile.get_attributes()
        trial_attrs = datafile.get_attributes(group="trials", trial=0)
        data_attrs = datafile.get_attributes(group="trials", trial=0, **test_set)
    print(data[0])
    print(data.dtype)
