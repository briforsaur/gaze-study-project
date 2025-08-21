# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from argparse import ArgumentParser
from pathlib import Path
from pupiltools.export import export_folder


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "folder_path",
        type=Path,
        help=(
            "Path to the the individual pupil recordings folders, i.e. "
            "<folder_path>/000, <folder_path>/001, etc."
        ),
    )
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the exported data."
    )
    parser.add_argument(
        "--filetype",
        default="csv",
        help="Export file type. Can be either 'csv' or 'hdf'",
    )
    parser.add_argument(
        "--experiment_log",
        type=Path,
        default=None,
        help="Log file from experiment describing which recordings to export. If not supplied, the entire folder of recordings will be exported.",
    )
    parser.add_argument(
        "--demographics_log",
        type=Path,
        default=None,
        help="Log file from the experiment describing the participant demographics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    export_folder(**vars(args))
