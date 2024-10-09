"""Functions to export data from Pupil Core recordings"""

import msgpack as mpk
import numpy as np
import pathlib
from collections.abc import Iterator
import csv
import json
from .data_structures import pupil_to_csv_fieldmap, get_csv_fieldnames, FieldMapKeyError
import h5py


ts_file_suffix = "_timestamps.npy"
data_file_suffix = ".pldata"


def export_folder(folder_path: pathlib.Path, output_path: pathlib.Path, experiment_log: pathlib.Path | None = None, filetype: str = "csv"):
    assert folder_path.exists()
    if experiment_log is not None:
        sub_folders = get_subfolders_from_log(folder_path, experiment_log)
    else:
        # Find all subfolders within the folder_path
        sub_folders = (subdir for subdir in folder_path.iterdir() if subdir.is_dir())
    if not output_path.exists():
        output_path.mkdir()
    topics = ("pupil",)
    if filetype == "csv":
        for sub_folder in sub_folders:
            export_data_csv(sub_folder, output_path, ("pupil",))
    elif filetype == "hdf":
        export_hdf()


def get_subfolders_from_log(folder_path: pathlib.Path, experiment_log: pathlib.Path) -> Iterator[pathlib.Path]:
    with open(experiment_log) as f:
        log_data = json.load(f)
    for item in log_data["trial_record"]:
        subfolder_path = folder_path / pathlib.Path(item["recording"])
        if subfolder_path.exists():
            yield subfolder_path
        else:
            print(f"Recording folder {subfolder_path} not found.")


def export_data_csv(
    folder_path: pathlib.Path, output_path: pathlib.Path, data_topics: tuple[str]
):
    """Export raw Pupil pldata and npy files to CSV"""
    world_ts_data = np.load(folder_path / "world_timestamps.npy")
    for topic in data_topics:
        data_file = folder_path / f"{topic}{data_file_suffix}"
        export_file = output_path / f"{folder_path.name}_{topic}_positions.csv"
        with open(export_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
            # Data processing pipeline using generators
            data = extract_data(data_file, t_start=world_ts_data[0], method="3d")
            timestamped_data = match_timestamps(data, world_ts_data)
            flattened_data = flatten_data(timestamped_data)
            writer.writeheader()
            writer.writerows(flattened_data)


def export_hdf():
    """Export raw Pupil pldata and npy files to HDF5 format"""
    pass


def extract_data(
    data_file: pathlib.Path, t_start: float, method: str = "3d"
) -> Iterator[dict]:
    """Extract Pupil data from pldata (msgpack) format"""
    with open(data_file, "rb") as f:
        unpacker = mpk.Unpacker(f, use_list=False)
        topic: str
        for topic, b_obj in unpacker:
            # Topic strings are of the form [top-level-topic].[eye_id].[method_id]
            if topic.split(".")[2] == method:
                data = mpk.unpackb(b_obj)
                if data["timestamp"] >= t_start:
                    yield data


def match_timestamps(data: Iterator[dict], ts_data: np.ndarray) -> Iterator[dict]:
    """Match data points to the nearest world frame index

    Adapted from:
    https://stackoverflow.com/a/8929827
    """
    for entry in data:
        target = entry["timestamp"]
        # Find the index where ts_data[idx-1] < target <= ts_data[idx]
        idx = ts_data.searchsorted(target)
        # If the closest index is 0 or the length of the list, clip to 1 or length - 1
        idx = np.clip(idx, 1, len(ts_data) - 1)
        left = ts_data[idx - 1]
        right = ts_data[idx]
        # Change idx to match the closer of the left or right index
        idx -= target - left < right - target
        entry.update({"world_index": int(idx)})
        yield entry


def flatten_data(data: Iterator[dict]) -> Iterator[dict]:
    """Turn an iterator of nested dicts into an iterator single-layer dicts"""
    for entry in data:
        output = {}
        for key, value in entry.items():
            try:
                output.update(make_flat(key, value, pupil_to_csv_fieldmap))
            except FieldMapKeyError:
                # Some entries in the pldata file don't map to a CSV field, can ignore
                pass
        yield output


def make_flat(key: str, value, fieldmap: dict) -> dict:
    """Turn a nested dict into a single-layer dict"""
    output = {}
    try:
        field_mapping = fieldmap[key]
    except KeyError:
        # If there is no corresponding key in the field mapping, raise to inform caller
        raise FieldMapKeyError
    # Different flattening behaviour using a Class Pattern match
    # https://docs.python.org/3/reference/compound_stmts.html#class-patterns
    # https://stackoverflow.com/a/77966563
    match field_mapping:
        case str():
            output.update({field_mapping: value})
        case list():
            output.update(zip(field_mapping, value))
        case dict():
            for subfieldkey in iter(field_mapping):
                # Use recursion to flatten the dict within a dict
                output.update(make_flat(subfieldkey, value[subfieldkey], field_mapping))
    return output

