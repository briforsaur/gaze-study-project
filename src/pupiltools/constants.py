from . import utilities


PARTICIPANTS = [f"P{utilities.make_digit_str(i, width=2)}" for i in range(1, 31)]
TASK_TYPES = ("action", "observation")
TS_FILE_SUFFIX = "_timestamps.npy"
DATA_FILE_SUFFIX = ".pldata"