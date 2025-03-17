from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
import logging

from pupiltools.utilities import make_digit_str, get_datetime
from pupiltools.data_analysis import get_timeseries, impute_missing_values
from pupiltools.constants import TASK_TYPES


logger = logging.getLogger(__name__)


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
    return parser.parse_args()


def main(data_filepath: Path, export_path: Path, t_max: float):
    participants = [f"P{make_digit_str(i, width=2)}" for i in range(1, 31)]
    variables = ("timestamp", "confidence", "diameter_3d")
    features = {}
    with h5py.File(data_filepath, mode='r') as f_root:
        for participant_id in participants:
            logger.info(f"Processing {participant_id}")
            feature_array = None
            for i_task, task in enumerate(TASK_TYPES):
                logger.info(f"    {task}")
                dataset: h5py.Dataset = f_root["/".join(["", participant_id, task])]
                task_data = dataset.fields(variables)[:]
                task_timeseries = get_timeseries(task_data["diameter_3d"], t_range=(1.0, t_max))
                task_id_array = np.full(shape=(task_timeseries.shape[0], 1), fill_value=i_task)
                labelled_features = np.concat((task_timeseries, task_id_array), axis=1)
                if feature_array is None:
                    feature_array = labelled_features
                else:
                    feature_array = np.concat((feature_array, labelled_features), axis=0)
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