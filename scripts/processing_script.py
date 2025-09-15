# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

import pupiltools.data_import as d_import
import pupiltools.data_analysis as da
import pupiltools.export as d_export
from pupiltools.utilities import make_digit_str, get_datetime

from argparse import ArgumentParser
import copy
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Args:
    data_path: Path
    export_path: Path
    filter_config_file: Path
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
        "filter_config_file",
        type=Path,
        help="Path to YAML file with filter configurations for each variable to be filtered.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Eye tracker confidence value below which data is discarded.",
        default=0.6,
    )
    return Args(**vars(parser.parse_args()))


def main(
    data_path: Path,
    export_path: Path,
    filter_config_file: Path,
    confidence_threshold: float,
):
    logging.basicConfig(level=logging.INFO)
    variables = ("timestamp", "confidence", "diameter_3d", "theta", "phi")
    hdf_path_info = {"group": "trials", "topic": "pupil"}
    participant_ids = ["P" + make_digit_str(i, 2) for i in range(1, 31)]
    processed_data = {}
    with open(filter_config_file, mode="r") as config_file:
        filter_configs: dict = yaml.load(config_file, Loader=yaml.Loader)
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
            for var_name, config in filter_configs.items():
                filter_config = da.FilterConfig(**config)
                p_data_array[var_name] = da.filter_signal(
                    p_data_array[var_name], filter_config=filter_config
                )
        processed_data.update({participant_id: copy.deepcopy(p_data_arraydict)})
    # Flatten the filter configs dict to be used as HDF file attributes
    flat_configs = {}
    for key, value in filter_configs.items():
        for subkey, subvalue in value.items():
            flat_configs.update({"_".join((key, subkey)): subvalue})
    if not export_path.exists():
        export_path.mkdir()
    d_export.save_processed_data(
        export_path
        / f"{get_datetime()}_processed_data_f{round(filter_configs['diameter_3d']['Wn']):.0f}.hdf5",
        processed_data,
        flat_configs,
    )


if __name__ == "__main__":
    args = get_args()
    main(
        data_path=args.data_path,
        export_path=args.export_path,
        filter_config_file=args.filter_config_file,
        confidence_threshold=args.confidence_threshold,
    )
