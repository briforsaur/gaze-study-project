from argparse import ArgumentParser
from pathlib import Path
from pupiltools.export import export_folder
from dataclasses import dataclass


@dataclass
class Args:
    recording_path: Path
    output_path: Path
    log_file_path: Path
    file_type: str
    demographics_path: Path


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument(
        "recording_path",
        type=Path,
        help=(
            "Path to the the individual pupil recordings folders, i.e. "
            "<recording_path>/000, <recording_path>/001, etc."
        ),
    )
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the exported data."
    )
    parser.add_argument(
        "--log_file_path",
        type=Path,
        default=None,
        help="Log file from experiment describing which recordings to export. If not supplied, the entire folder of recordings will be exported.",
    )
    parser.add_argument(
        "--file_type",
        default="csv",
        help="Export file type. Can be either 'csv' or 'hdf'",
    )
    parser.add_argument(
        "--demographics_path",
        type=Path,
        default=None,
        help="Log file from the experiment describing the participant demographics.",
    )
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    args = get_args()
    export_folder(
        args.recording_path,
        args.output_path,
        args.log_file_path,
        args.file_type,
        args.demographics_path,
    )
