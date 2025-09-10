# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

import argparse
import h5py
from numpy.lib.npyio import NpzFile
import numpy as np
import os
from pathlib import Path
from collections.abc import Iterable
from typing import cast
from .aliases import pupil_datatype, gaze_datatype, AttributesType, RawParticipantDataType, ResampledParticipantDataType
from .utilities import make_digit_str


class GazeDataFile:
    """Object for handling gaze study HDF5 data files

    This class serves as a user interface for getting data and attributes from an HDF5
    file with a specific structure.

    Expects an HDF5 file in the following structure::
        
        root/
        |-- trials/
        |   |-- 000/
        |   |   |-- pupil_eye0_3d
        |   |   |-- pupil_eye1_3d
        |   |-- 001/
        |   |   |-- pupil_eye0_3d
        |   |   |-- pupil_eye1_3d

    Can be used with a context manager (``with`` statement) to automatically
    close the file when no longer needed or when an exception occurs.

    Examples
    --------

    Opening a file with a context manager::

        with GazeDataFile("P01.hdf", mode="r") as datafile:
            data = datafile.get_data(
                group="trials",
                trial=10,
                topic="pupil",
                eye=0,
                variables=["timestamp", "world_index", "diameter_3d"],
            )
            print(data[0])
    
    """

    def __init__(self, file: str | bytes | os.PathLike, mode: str = 'r'):
        """Instantiate an instance of a GazeDataFile object
        
        Parameters
        ----------
        file: str | bytes | pathlike
            Path or file-like object to the hdf5 file
        mode: str
            File opening mode. Only supports modes supported by h5py's File object:

            ``'r'``
                Readonly, file must exist (default)
            ``'r+'``
                Read/write, file must exist
            ``'w'``
                Create file, truncate if exists
            ``'w-'`` or ``'x'``
                Create file, fail if exists
            ``'a'``
                Read/write if exists, create otherwise
        """
        self.file = file
        self.hdf_root = h5py.File(file, mode=mode)
        self.n_trials = len(self.hdf_root["trials"].keys()) # type: ignore

    def get_data(
        self,
        group: str = "trials",
        trial: int = 0,
        topic: str = "pupil",
        eye: int = -1,
        variables: str | Iterable[str] = "all",
    ) -> np.ndarray:
        """Get a numpy array from a gaze study dataset.

        Extract a variable or set of variables from an HDF5 file containing gaze study 
        datasets. 

        Parameters
        ----------
        group: str, default="trials"
            The data group, can be either ``"trials"`` or [TODO] ``"calibrations"``.
        trial: int, default=0
            The trial or calibration number.
        topic: str, default="pupil"
            The data topic, normally either ``"pupil"`` or ``"gaze"``.
        eye: {0, 1}, default=0
            The eye number. ``0`` for participant's right, ``1`` for left.
        variables: str | list[str], default="all"
            The variables to extract from the dataset. Can either be a single string 
            representing a single variable's name, or a list of strings representing the 
            names of multiple variables. The variable names can be any top-level 
            variable name in the dataset, for example ``"timestamp"`` or ``"norm_pos"``.
            If ``'all'``, all variables in the dataset are returned.
        
        Returns
        -------
        numpy.ndarray
            A structured array containing the requested variables within the dataset. 
            The numpy datatype depends on the topic and variables requested.

            See :py:type:`pupiltools.aliases.pupil_datatype` and 
            :py:type:`pupiltools.aliases.gaze_datatype`.
        """
        if group == "calibrations":
            raise NotImplementedError("Calibrations retrieval not yet implemented.")
        datapath = get_path(group=group, trial=trial, topic=topic, eye=eye)
        dataset = self.hdf_root[datapath]
        assert isinstance(dataset, h5py.Dataset) # Assert appeases pylance
        if isinstance(variables, str) and variables == "all":
            # Preallocate an array and read the dataset in directly to avoid an
            # intermediate copy of the entire dataset
            match topic:
                case "pupil":
                    dtype = pupil_datatype
                case "gaze":
                    dtype = gaze_datatype
                case _:
                    raise NotImplementedError(f"No known numpy datatype corresponding to {topic}")
            data = np.empty(shape=dataset.shape, dtype=dtype)
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
    ) -> AttributesType:
        """Get the attributes of a member of the HDF File

        HDF file members (groups, datasets) have attributes that specify metadata about
        that member. This function gets the attributes from a specified member in the 
        file.

        The path to the member is generated automatically from the parameters. See
        :py:func:`get_path`.

        Parameters
        ----------
        group: str, default=""
            The HDF file top-level group.
        trial: int, default=-1
            The trial group number.
        topic: str, default=""
            The dataset topic.
        eye: int, default=-1
            The dataset eye id.
        
        Returns
        -------
        :py:type:`pupiltools.aliases.AttributesType`
            A dictionary of attributes of the given group or dataset.
        """
        path = get_path(group=group, trial=trial, topic=topic, eye=eye)
        hdf_member = self.hdf_root[path]
        assert isinstance(hdf_member, h5py.Group | h5py.Dataset) # Assert appeases pylance
        # Need to use dict() to avoid shallow copy that is left dangling on file close
        return dict(hdf_member.attrs.items())
    
    def close(self):
        """Close the HDF file connection. Called by context manager automatically."""
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
) -> str:
    """Generate an internal HDF path to the desired group or dataset
    
    All parameters are empty or invalid by default. If left as defaults, they will
    not be included in the path.

    If ``topic`` is not an empty string, a dataset name is generated from ``topic`` and
    ``eye`` by :py:func:`get_dataset_name`.

    Parameters
    ----------
    group: str, default=""
        The top-level data group in the file. Reserved names: ``'root'`` and ``'/'``
        result in the root group being returned and all other parameters are ignored.
    trial: int, default=-1
        The trial number. Must be greater than or equal to 0.
    topic: str, default=""
        The dataset topic.
    eye: int, default=-1
        The dataset eye id.

    Returns
    -------
    str
        A path to the desired group or dataset in the HDF file.

    Examples
    --------
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
    >>> get_path(group="trials", trial=0, topic="pupil", eye=0)
    '/trials/000/pupil_eye0_3d'
    >>> get_path(group="trials", trial=0, topic="gaze")

    """
    if group in ["root", "/"]:
        path = "/"
    else:
        trial_str = make_digit_str(trial, width=3) if trial >= 0 else ""
        dataset_name = get_dataset_name(topic, eye)
        # Creating a list of non-empty path members
        pathlist = [var for var in [group, trial_str, dataset_name] if var]
        path = "/" + "/".join(pathlist)
    return path


def get_dataset_name(topic: str = "", eye: int = -1) -> str:
    """Generate a dataset name string from a topic and eye id

    Generates a dataset name from the parameters. The default behaviour results in an
    empty string so that this function can be used in group lookup as well.

    Parameters
    ----------
    topic: str, default=""
        The data topic name.
    eye: int, default=-1
        The eye id for when the topic is ``'pupil'``. Acceptable values are 0 or 1.
    
    Returns
    -------
    str
        The dataset name, if a non-empty topic string is given, or an empty string.
    
    Raises
    ------
    ValueError
        If the ``topic`` is ``'pupil'`` and the ``eye`` parameter is not ``0`` or ``1``.

    Examples
    --------
    >>> get_dataset_name()
    ''
    >>> get_dataset_name("pupil", 0)
    'pupil_eye0_3d'
    >>> get_dataset_name("gaze")
    'gaze'
    """
    eye_str = ""
    method = ""
    if topic == "pupil":
        method = "3d"
        if eye in [0, 1]:
            eye_str = f"eye{eye}"
        else:
            raise ValueError("Pupil topic requires an eye index of 0 or 1.")
    # Making the dataset name out of nonempty string parts
    dataset_name_list = [part for part in [topic, eye_str, method] if part]
    return "_".join(dataset_name_list)


def get_raw_participant_data(file: str | bytes | os.PathLike, group: str = "trials", topic: str = "pupil", variables: str | Iterable[str] = "all") -> tuple[RawParticipantDataType, AttributesType]:
    """Get raw participant data from an HDF File for all trials
    
    Parameters
    ----------
    file: str | bytes | os.PathLike
        HDF file path, or an open file.
    group: str, default="trials"
        HDF top-level data group.
    topic: str, default="pupil"
        Dataset topic.
    variables: str | Iterable[str], default="all"
        Variable(s) to export. Can be a single string with a variable name or "all", in
        which case all variables will be imported, or a list of strings of variable 
        names if only a subset of variables is desired.
    
    Returns
    -------
    participant_data: :py:type:`pupiltools.aliases.RawParticipantDataType`
        A list of metadata and data for each trial.
    participant_metadata: :py:type:`pupiltools.aliases.AttributesType`
        A dictionary of metadata describing the participant's demographics.
    """
    participant_data = []
    with GazeDataFile(file, mode='r') as datafile:
        participant_metadata = datafile.get_attributes()
        for i_trial in range(datafile.n_trials):
            attr = datafile.get_attributes(group=group, trial=i_trial)
            data = _get_eye_data(datafile, variables=variables, trial=i_trial, group=group, topic=topic)
            participant_data.append({"attributes": attr, "data": data})
    return participant_data, participant_metadata


def get_resampled_participant_data(file: str | bytes | os.PathLike, group: str = "trials", topic: str = "pupil", variables: str | Iterable[str] = "all") -> tuple[ResampledParticipantDataType, dict]:
    """Get resampled participant data from an HDF File
    
    Effectively a wrapper for :py:func:`get_raw_participant_data` with a different 
    return type. The only difference in the return type is that the numpy arrays in
    the data are combined for both eyes, i.e. ``N x n_variables x 2``.
    """
    # Just a wrapper for get_raw_participant data with the correct return type hint
    hdf_path_info = {"group": group, "topic": topic}
    participant_data, participant_metadata = get_raw_participant_data(file=file, variables=variables, **hdf_path_info)
    participant_data = cast(ResampledParticipantDataType, participant_data)
    return participant_data, participant_metadata


def _get_eye_data(datafile: GazeDataFile, variables: str | Iterable[str], trial: int, group: str = "trials", topic: str = "pupil", eyes: tuple[int,...] = (0, 1)) -> np.ndarray | list[np.ndarray]:
    """Get raw or resampled data from an HDF file

    Parameters
    ----------
    datafile: GazeDataFile
        An open GazeDataFile. This function does not handle context management.
    variables: variables: str | Iterable[str]
        A single string or list of strings of variable names. See 
        :py:meth:`GazeDataFile.get_data` for details.
    trial: int
        The trial number. Must be greater than or equal to 0.
    group: str, default="trials"
        The top-level group in the HDF file.
    topic: str, default="pupil"
        Dataset topic.
    eyes: tuple[int,...], default=(0, 1)
        Eyes to extract from the dataset. Only valid for raw data files, which store
        the data for each eye in separate datasets. For a resampled data file, this
        parameter has no effect.
    
    Returns
    -------
    np.ndarray | list[np.ndarray]
        A numpy array (if the file is the resampled data) or list of numpy arrays (if 
        the file is the raw data). The two types of data files store eye data slightly
        differently: raw data HDF files have the eye data stored in two separate 
        datasets, one for each eye, whereas the resampled data HDF files have the eye
        data stored in a single, multidimensional numpy array.
    """
    group_path = get_path(group=group, trial=trial)
    group_obj: h5py.Group = datafile.hdf_root[group_path] # type: ignore
    data = []
    # Check to see if the eye data is split across 2 datasets (raw data) or in one (resampled)
    for eye in eyes:
        if get_dataset_name(topic=topic, eye=eye) in group_obj:
            data.append(datafile.get_data(group=group, trial=trial, topic=topic, eye=eye, variables=variables))
    # If the data list is still empty, then the eyes must be in a single dataset
    if not data:
        data = datafile.get_data(group=group, trial=trial, topic=topic, eye=-1, variables=variables)
    return data


def get_class_data(class_data_file: NpzFile, ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            group_labels = np.concat((group_labels, group_label)) # type: ignore
    return labelled_feature_data[:, :-1], labelled_feature_data[:, -1].astype(np.int64), group_labels # type: ignore


def _main(data_file_path: Path, topic: str, all_vars: bool):
    test_set = {"topic": topic}
    match topic:
        case "pupil":
            test_set["eye"] = 0 # type:ignore
            variables = ["timestamp", "world_index", "diameter_3d"]
        case "gaze":
            variables = ["timestamp", "world_index", "norm_pos"]
        # case _:
        #     raise NotImplementedError
    if all_vars:
        variables = "all"
    with GazeDataFile(data_file_path, mode="r") as datafile:
        data = datafile.get_data(variables=variables, **test_set) # type:ignore
        root_attrs = datafile.get_attributes()
        trial_attrs = datafile.get_attributes(group="trials", trial=0)
        data_attrs = datafile.get_attributes(group="trials", trial=0, **test_set) # type:ignore
    print(data[0])
    print(data.dtype)
    print(root_attrs)
    print(trial_attrs)
    print(data_attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file_path", type=Path, help="Path to the HDF5 data file")
    parser.add_argument("topic", type=str, help="Dataset topic")
    parser.add_argument("--all_vars", action="store_true")
    args = parser.parse_args()
    _main(**vars(args))
