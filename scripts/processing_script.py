import pupiltools.data_import as d_import
import pupiltools.data_analysis as da
from pupiltools.utilities import make_digit_str

from argparse import ArgumentParser
import copy
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Args:
    data_path: Path
    export_path: Path
    confidence_threshold: float


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "data_path", type=Path, help="Path to directory containing the HDF files."
    )
    parser.add_argument(
        "export_path", type=Path, help="Path to directory to save the processed data."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Eye tracker confidence value below which data is discarded.",
        default=0.6,
    )
    return parser.parse_args()


def main(data_path: Path, export_path: Path, confidence_threshold: float):
    logging.basicConfig(level=logging.INFO)
    variables = ("timestamp", "confidence", "diameter_3d", "theta", "phi")
    hdf_path_info = {"group": "trials", "topic": "pupil", "method": "3d"}
    participant_ids = ["P" + make_digit_str(i, 2) for i in range(1, 31)]
    processed_data = {}
    for participant_id in participant_ids:
        logger.info(f"Processing {participant_id}")
        file_path = data_path / f"{participant_id}.hdf5"
        participant_data, _ = d_import.get_resampled_participant_data(
            file_path, variables=variables, **hdf_path_info
        )
        p_data_arraydict = da.convert_to_array(participant_data)
        for p_data_array in p_data_arraydict.values():
            da.remove_low_confidence(p_data_array, confidence_threshold)
            da.interpolate_nan(p_data_array)
            da.normalize_pupil_diameter(p_data_array)
        processed_data.update(
            {participant_id: copy.deepcopy(p_data_arraydict["action"])}
        )
    # TODO: Save the processed data to a file


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
