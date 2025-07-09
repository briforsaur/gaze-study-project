from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
import logging

from pupiltools.utilities import make_digit_str, get_datetime
from pupiltools.data_analysis import get_features, impute_missing_values


logger = logging.getLogger(__name__)


TASK_TYPES = ("action", "observation")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "data_filepath", type=Path, help="Path to the HDF file containing processed data."
    )
    parser.add_argument(
        "export_path", type=Path, help="Path to directory to save the feature data."
    )
    return parser.parse_args()


def main(data_filepath: Path, export_path: Path):
    participants = [f"P{make_digit_str(i, width=2)}" for i in range(1, 31)]
    variables = ("timestamp", "confidence", "diameter_3d")
    features = {}
    with h5py.File(data_filepath, mode='r') as f_root:
        for participant_id in participants:
            feature_array = None
            for i_task, task in enumerate(TASK_TYPES):
                dataset = f_root["/".join(["", participant_id, task])]
                assert isinstance(dataset, h5py.Dataset)
                task_data = dataset.fields(variables)[:]
                task_features = get_features(task_data["diameter_3d"], t_range=(1.0, np.inf))
                task_id_array = np.full(shape=(task_features.shape[0], 1), fill_value=i_task)

                labelled_features = np.concat((task_features, task_id_array), axis=1)
                if feature_array is None:
                    feature_array = labelled_features
                else:
                    feature_array = np.concat((feature_array, labelled_features), axis=0)
            assert isinstance(feature_array, np.ndarray)
            feature_array = impute_missing_values(feature_array)
            features.update({participant_id: feature_array})
    if not export_path.exists():
        export_path.mkdir()
    export_filepath = export_path / f"{get_datetime()}_feature_data.npz"
    np.savez(file=export_filepath, **features)
                


if __name__=="__main__":
    args = get_args()
    now = get_datetime()
    logging.basicConfig(filename=f"logs/{now}_feature_extract.log", filemode='w', level=logging.INFO)
    main(**vars(args))