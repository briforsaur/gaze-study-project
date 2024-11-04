from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

from pupiltools.utilities import fix_datetime_string, make_digit_str


_RECORD_NAMES = ("trial_record", "calibration_record")


def get_args():
    parser = ArgumentParser(
        description=("Fix experiment data that was split across multiple recording "
                     "sessions")
    )
    parser.add_argument(
        "log_paths", 
        nargs=2, 
        type=Path, 
        help=("A list of paths to two log files. The second file will be appended to "
              "the first.")
    )
    parser.add_argument(
        "output_filepath", type=Path, help="The destination for the complete log file."
    )
    parser.add_argument(
        "trial_offset", 
        type=int, 
        help=("The amount to offset the trial and calibration numbers from the second "
              "log")
    )
    parser.add_argument(
        "--trim_both",
        action="store_true",
        help=("Flag to indicate if the last trial of both logs should be trimmed, "
              "otherwise only the first log is trimmed.")
    )
    parser.add_argument(
        "--adjust_rec_num", 
        action="store_true", 
        help="TODO: Flag to indicate if the recording numbers must be adjusted as well."
    )
    return parser.parse_args()


def get_logs(log_paths: list[Path]) -> list[dict]:
    """Load log data from a list of JSON files"""
    logs = []
    for log_path in log_paths:
        with open(log_path, "r") as logfile:
            logs.append(json.load(logfile))
    return logs


def get_time_offset(time_strs: list[str]) -> int:
    """Get the time offset in seconds between two ISO datetimes with typos"""
    datetimes = []
    for time_str in time_strs:
        fixed_time = fix_datetime_string(time_str)
        datetimes.append(datetime.fromisoformat(fixed_time))
    t_offset: timedelta = datetimes[1] - datetimes[0]
    return t_offset.seconds


def offset_record(record: list[dict], trial_offset: int, t_offset: int, calibration: bool = False):
    """Adjust all trial numbers and timestamps of a trial or calibration record"""
    if calibration:
        timestamp_names = ("t_start", "t_end")
    else:
        timestamp_names = ("t_start", "t_instruction", "t_stop")
    for logitem in record:
        # Only "trial_record" objects have a "trial" key, "calibration_records" do not
        if not calibration:
            logitem["trial"] += trial_offset
        for timestamp in timestamp_names:
            logitem[timestamp] += t_offset


def remove_last_records(log: dict[str, list]):
    """Remove the last trial record and calibration record in a log"""
    for record in _RECORD_NAMES:
        log[record].pop()


def append_logs(logs: list[dict[str, list]]):
    """Append the trials and calibrations of two logs, return a new log"""
    complete_log = deepcopy(logs[0])
    for record in _RECORD_NAMES:
        # Concatenating the lists of record items
        complete_log[record] += logs[1][record]
    return complete_log


def save_log(output_filepath, log):
    """Save log to a json file"""
    with open(output_filepath, mode="w", encoding="utf-8") as log_file:
        json.dump(log, log_file, indent=4)


def main(log_paths: list[Path], output_filepath: Path, trial_offset: int, trim_both: bool = False, adjust_rec_num: bool = False):
    """Fix experiment data that was split across multiple recording sessions
    
    Parameters
    ----------
    log_paths: list[pathlib.Path]
        A list of paths to two log files. The second file will be appended to the first.
    output_filepath: pathlib.Path
        The destination for the complete log file.
    trial_offset: int
        The amount to offset the trial and calibration numbers from the second log
    trim_logs: list[bool] = [False, False]
        List of booleans (or 0, 1) to indicate which log files to trim the final trial 
        and calibration.
    adjust_rec_num: bool = False
        TODO: Flag to indicate if the recording numbers must be adjusted as well.
    """
    # For P20:
    logs = get_logs(log_paths)
    # Get the datetime strings for each log
    time_strs = [log["header"]["date"] for log in logs]
    t_offset = get_time_offset(time_strs)
    # Take the trials from 2nd log:
    #   adjust all the trial numbers by 30
    #   adjust all the start, end, and instruction times by the time offset
    for record in _RECORD_NAMES:
        calib = "calibration" in record
        offset_record(logs[1][record], trial_offset, t_offset, calibration=calib)
    # Remove the last trial and calibration from both logs
    for i, log in enumerate(logs):
        if trim_both or i == 0:
            remove_last_records(log)
    # Append log 2 to log 1
    complete_log = append_logs(logs)
    # Save new log file in the output folder
    save_log(output_filepath, complete_log)
    # For P18
    # Recording numbers need to be adjusted
    #   in logs
    #   in files
    # Recordings need to be copied into a single folder
    # The last calibration and trial DO NOT need to be removed from the 2nd set


if __name__=="__main__":
    args = get_args()
    main(**vars(args))