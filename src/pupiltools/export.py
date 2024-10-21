"""Functions to export data from Pupil Core recordings"""

import msgpack as mpk
import numpy as np
import pathlib
from collections.abc import Iterator, Container
import csv
import json
from .data_structures import pupil_to_csv_fieldmap, get_csv_fieldnames, FieldMapKeyError
from .aliases import pupil_datatype
from .utilities import fix_datetime_string, make_digit_str
import h5py


ts_file_suffix = "_timestamps.npy"
data_file_suffix = ".pldata"


def export_folder(folder_path: pathlib.Path, output_path: pathlib.Path, experiment_log: pathlib.Path | None = None, filetype: str = "csv", demographics_log: pathlib.Path | None = None):
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
            export_data_csv(sub_folder, output_path, topics)
    elif filetype == "hdf":
        metadata = get_metadata(experiment_log, demographics_log)
        export_hdf(folder_path, output_path, sub_folders, metadata)


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


def export_hdf(folder_path: pathlib.Path, output_path: pathlib.Path, sub_folders: Iterator[pathlib.Path], metadata: dict):
    """Export raw Pupil pldata and npy files to HDF5 format"""
    topic = "pupil"
    export_file = output_path / f"{folder_path.name}.hdf5"
    with h5py.File(export_file, 'w') as f_root:
        f_root.attrs.update(metadata["header"])
        trials_group = f_root.create_group("trials")
        for trial_num, sub_folder in enumerate(sub_folders):
            trial_group = trials_group.create_group(make_digit_str(trial_num))
            trial_group.attrs.update(metadata["trial_record"][trial_num])
            world_ts_data = np.load(sub_folder / "world_timestamps.npy")
            data_file = sub_folder / f"{topic}{data_file_suffix}"
            data = extract_data(data_file, world_ts_data[0], method="3d")
            timestamped_data = match_timestamps(data, world_ts_data)
            grouped_data = {}
            for data_entry in timestamped_data:
                method = data_entry["topic"].split(".")[-1]
                dataset_name = f"{topic}_eye{data_entry['id']}_{method}"
                if dataset_name not in grouped_data:
                    grouped_data[dataset_name] = []
                grouped_data[dataset_name].append(data_entry)
            for dataset_name, dataset_values in grouped_data.items():
                dataset_values = pupildata_to_numpy(dataset_values)
                dataset = trial_group.create_dataset(dataset_name, data=dataset_values)
                name_parts = dataset_name.split("_")
                dataset_metadata = {
                    "topic": name_parts[0],
                    "eye id": name_parts[1][-1],
                    "method": "pye3d 0.3.0 real-time",
                }
                dataset.attrs.update(dataset_metadata)


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


def nested_dict_to_tuple(nested_dict: dict, datatype: np.dtype = pupil_datatype):
    data_list = []
    for field in datatype.names:
        if not isinstance(nested_dict[field], Container):
            data_list.append(nested_dict[field])
        elif isinstance(nested_dict[field], list):
            data_list.append(tuple(nested_dict[field]))
        elif isinstance(nested_dict[field], dict):
            data_list.append(
                nested_dict_to_tuple(nested_dict[field], datatype.fields[field][0])
            )
    return tuple(data_list)


def pupildata_to_numpy(pupil_data: list) -> np.ndarray:
    data = []
    for data_entry in pupil_data:
        data.append(nested_dict_to_tuple(data_entry))
    return np.array(data, dtype=pupil_datatype)


def get_metadata(experiment_log: pathlib.Path, demographics_log: pathlib.Path) -> dict:
    experiment_metadata: dict[str, dict] = load_json_log(experiment_log)
    participant_id = experiment_metadata["header"]["participant_id"]
    demographic_metadata = load_json_log(demographics_log)
    demographic_metadata = demographic_metadata[participant_id]
    experiment_metadata["header"].update(demographic_metadata)
    session_datetime = experiment_metadata["header"]["date"]
    session_datetime = fix_datetime_string(session_datetime)
    experiment_metadata["header"]["date"] = session_datetime
    return experiment_metadata


def load_json_log(log_file_path: pathlib.Path) -> dict:
    with open(log_file_path, "r") as f:
        log_data = json.load(f)
    return log_data
