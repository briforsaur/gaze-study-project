import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from .data_structures import pupil_datatype
from .utilities import make_digit_str


def get_data(
    file_path: str | bytes | os.PathLike,
    group: str = "trials",
    trial: int = 0,
    topic: str = "pupil",
    eye: int = 0,
    method: str = "3d",
    variables: str | list[str] = "all",
) -> np.ndarray:
    """Get a numpy array from a gaze study dataset.

    Extract a variable or set of variables from an HDF5 file containing gaze study 
    datasets. Expects an HDF5 file in the following structure:

        root/
        |-- trials/
        |   |-- 000/
        |   |   |-- pupil_eye0_3d
        |   |   |-- pupil_eye1_3d
        |   |-- 001/
        |   |   |-- pupil_eye0_3d
        |   |   |-- pupil_eye1_3d

    Parameters
    ----------
    file: str | bytes | pathlike
        Path or file-like object to the hdf5 file
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
        names of multiple variables. The variable names can be any top-level variable 
        name in the dataset, for example "timestamp" or "norm_pos".
    
    Returns
    -------
    numpy.ndarray
        A structured array containing the requested variables within the dataset. 
    """
    trial_str = make_digit_str(trial, width=3)
    dataset_name = "_".join([topic, f"eye{eye}", method])
    datapath = "/".join([group, trial_str, dataset_name])
    with h5py.File(file_path, mode="r") as hdf_root:
        dataset: h5py.Dataset = hdf_root[datapath]
        if isinstance(variables, str) and variables == "all":
            # Preallocate an array and read the dataset in directly to avoid an
            # intermediate copy of the entire dataset
            data = np.empty(shape=dataset.shape, dtype=pupil_datatype)
            dataset.read_direct(data)
        else:
            data = np.empty(
                shape=(dataset.shape[0], len(variables)),
                dtype=pupil_datatype[variables],
            )
            data = dataset.fields(variables)[:]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file_path", type=Path, help="Path to the HDF5 data file")
    args = parser.parse_args()
    data = get_data(
        args.data_file_path, variables=["timestamp", "world_index", "diameter_3d"]
    )
    print(data[0])
    print(data.dtype)
