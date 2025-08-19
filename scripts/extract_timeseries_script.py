# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path
import h5py
import numpy as np
import logging

from pupiltools.utilities import make_digit_str, get_datetime
from pupiltools.data_analysis import get_all_timeseries, calc_index_from_time, impute_missing_values
from pupiltools.constants import TASK_TYPES


logger = logging.getLogger(__name__)
DEFAULT_VARS = ("diameter_3d",)
N_TRIALS = 120


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the HDF file containing processed data."
    )
    parser.add_argument(
        "export_path", type=Path, help="Path to directory to save the feature data."
    )
    parser.add_argument(
        "t_max", type=float, help="Maximum time value for extracted timeseries"
    )
    parser.add_argument("--N_trials", type=int, default=N_TRIALS, help="Total number of trials per experiment across all tasks.")
    parser.add_argument("--variables", type=str, nargs='*', help="Variables other than timestamp and confidence to include in the exported timeseries.", default=DEFAULT_VARS)
    return parser.parse_args()


def main(data_filepath: Path, export_path: Path, t_max: float, N_trials: int = N_TRIALS, variables: Iterable[str] = DEFAULT_VARS):
    participants = [f"P{make_digit_str(i, width=2)}" for i in range(1, 31)]
    full_variables = ("timestamp", "confidence", *variables)
    features = {}
    with h5py.File(data_filepath, mode='r') as f_root:
        for participant_id in participants:
            logger.info(f"Processing {participant_id}")
            feature_array = np.zeros((0,0), dtype=np.float64)
            trial_indices = [0, 0]
            for i_task, task in enumerate(TASK_TYPES):
                logger.info(f"    {task}")
                dataset = f_root["/".join(["", participant_id, task])]
                assert isinstance(dataset, h5py.Dataset) # Assert to appease pylance
                task_data = dataset.fields(full_variables)[:]
                i_range = calc_index_from_time(task_data.shape[0], t_range=(1.0, t_max))
                task_timeseries = get_all_timeseries(task_data, i_range, variables)
                task_id_array = np.full(shape=(task_timeseries.shape[0], 1), fill_value=i_task)
                labelled_features = np.concat((task_timeseries, task_id_array), axis=1)
                if feature_array.shape == (0,0):
                    # Allocating the max array size, although due to pruning the full 
                    # array will likely not be used
                    feature_array = np.zeros((N_trials, labelled_features.shape[1]))
                trial_indices[1] = trial_indices[1] + labelled_features.shape[0]
                feature_array[trial_indices[0]:trial_indices[1], :] = labelled_features
                trial_indices[0] = trial_indices[1]
            # Removing the unused rows from the feature array
            n_trials_not_pruned = trial_indices[1]
            feature_array = feature_array[:n_trials_not_pruned, :]
            features.update({participant_id: feature_array})
    if not export_path.exists():
        export_path.mkdir()
    export_filepath = export_path / f"{get_datetime()}_timeseries_data.npz"
    np.savez(file=export_filepath, **features)
                


if __name__=="__main__":
    args = get_args()
    now = get_datetime()
    scriptname = Path(__file__).stem
    logpath = Path(f"./logs/{scriptname}")
    if not logpath.exists():
        logpath.mkdir()
    logging.basicConfig(filename=logpath /f"{now}_timeseries.log", filemode='w', level=logging.INFO)
    logger.info(__file__)
    logger.info(now)
    logger.info(vars(args))
    main(**vars(args))