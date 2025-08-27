from . import utilities


PARTICIPANTS = [f"P{utilities.make_digit_str(i, width=2)}" for i in range(1, 31)]
"""List of participant IDs from P01 to P30"""

TASK_TYPES = ("action", "observation")
"""Tuple of task types - action and observation"""

TS_FILE_SUFFIX = "_timestamps.npy"
"""Suffix for timestamp files, _timestamps.npy"""

DATA_FILE_SUFFIX = ".pldata"
"""Suffix for pupil/gaze data files, .pldata"""