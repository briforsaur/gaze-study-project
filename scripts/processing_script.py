import pupiltools.data_import as d_import
import pupiltools.data_analysis as da
import pupiltools.export as d_export
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
    f_c: float


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
    parser.add_argument(
        "--f_c",
        type=float,
        help="Low-pass filter cutoff frequency.",
        default=5,
    )
    return Args(**vars(parser.parse_args()))


def main(data_path: Path, export_path: Path, confidence_threshold: float, f_c: float = 5):
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
            p_data_array["diameter_3d"] = da.filter_signal(
                p_data_array["diameter_3d"], f_s=100, f_c = f_c
            )
        processed_data.update(
            {participant_id: copy.deepcopy(p_data_arraydict)}
        )
    if not export_path.exists():
        export_path.mkdir()
    d_export.save_processed_data(export_path / f"processed_data_f{round(f_c):.0f}.hdf5", processed_data)


if __name__ == "__main__":
    args = get_args()
    main(data_path=args.data_path, export_path=args.export_path, confidence_threshold=args.confidence_threshold, f_c=args.f_c)
