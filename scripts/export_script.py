import argparse
import pathlib
import pupiltools.export as export
from dataclasses import dataclass, fields


@dataclass
class Args:
    recording_path: pathlib.Path
    output_path: pathlib.Path
    log_file_path: pathlib.Path
    annotations: bool


def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recording_path",
        type=pathlib.Path,
        help=(
            "Path to the the individual pupil recordings folders, i.e. "
            "<recording_path>/000, <recording_path>/001, etc."
        ),
    )
    parser.add_argument(
        "output_path", type=pathlib.Path, help="Directory to save the exported data."
    )
    parser.add_argument(
        "--log_file_path", type=pathlib.Path, default=None,
        help="Log file from experiment describing which recordings to export. If not supplied, the entire folder of recordings will be exported."
    )
    parser.add_argument(
        "-a",
        "--annotations",
        action="store_true",
        help="export annotations (currently does nothing)",
    )
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    args = get_args()
    export.export_folder(args.recording_path, args.output_path, args.log_file_path)
