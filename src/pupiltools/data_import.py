import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from typing import TypeAlias
from .aliases import pupil_datatype, AttributesType, RawParticipantDataType
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
        eye: int = 0,
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
    """
    if trial >= 0:
        trial_str = make_digit_str(trial, width=3)
    else:
        trial_str = ""
    if topic and method and (eye in [0, 1]):
        dataset_name = "_".join([topic, f"eye{eye}", method])
    else:
        dataset_name = ""
    # Creating a list of non-empty path members
    pathlist = [var for var in [group, trial_str, dataset_name] if var]
    if group in ["root", "/"]:
        path = "/"
    else:
        path = "/" + "/".join(pathlist)
    return path


def get_raw_participant_data(file: str | bytes | os.PathLike, group: str = "trials", topic: str = "pupil", method: str = "3d", variables: str | list[str] = "all") -> tuple[RawParticipantDataType, dict]:
    """Get raw participant data from an HDF File"""
    hdf_path_info = {"group": group, "topic": topic, "method": method}
    participant_data = []
    with GazeDataFile(file, mode='r') as datafile:
        participant_metadata = datafile.get_attributes()
        for i_trial in range(datafile.n_trials):
            attr = datafile.get_attributes(trial=i_trial, **hdf_path_info)
            data = []
            for eye in (0, 1):
                data.append(datafile.get_data(trial=i_trial, eye=eye, variables=variables, **hdf_path_info))
            participant_data.append({"attributes": attr, "data": data})
    return participant_data, participant_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file_path", type=Path, help="Path to the HDF5 data file")
    args = parser.parse_args()
    test_set = {"topic": "pupil", "eye": 0, "method": "3d"}
    with GazeDataFile(args.data_file_path, mode="r") as datafile:
        data = datafile.get_data(
            variables=["timestamp", "world_index", "diameter_3d"]
        )
        root_attrs = datafile.get_attributes()
        trial_attrs = datafile.get_attributes(group="trials", trial=0)
        data_attrs = datafile.get_attributes(group="trials", trial=0, **test_set)
    print(data[0])
    print(data.dtype)
