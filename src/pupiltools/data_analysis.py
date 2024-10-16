from numpy import typing as npt
import numpy as np
from .aliases import RawParticipantDataType, TrialDataType


def calc_deltas(array: np.ndarray) -> np.ndarray:
    """Calculates the differences between adjacent values in an array
    
    For an input array of length N, the function returns an N-1 array
    of the difference between elements n+1 and n, from n = 0 to N-2"""
    return array[1:] - array[:-1]


def get_time_deltas(participant_data: RawParticipantDataType) -> list[np.ndarray]:
    """Calculates the time differences between all samples in a dataset

    Parameters
    ----------
    participant_data: list[np.ndarray]
        A list of NumPy ndarrays, where each array is the time vector for a specific
        trial and eye dataset.
    
    Returns
    -------
    list[np.ndarray]
        A list of NumPy ndarrays, where each item in the list corresponds to a
        trial number and eye number (0 is trial 0, eye 0, 1 is trial 0, eye 1,
        2 is trial 1, eye 0, etc.). Each array is the time difference between
        adjacent samples for that trial and eye.
    """
    t_deltas = []
    trial: TrialDataType
    for trial in participant_data:
        for eye in (0,1):
            t = trial["data"][eye]["timestamp"]
            dt = calc_deltas(t)
            t_deltas.append(dt)
    return t_deltas


def get_time_delta_stats(t_deltas: list[np.ndarray]) -> dict[str, np.float64]:
    """Calculates basic statistics for the time differences between samples
    
    Parameters
    ----------
    t_deltas: list[np.ndarray]
        A list of NumPy ndarrays, where each item in the list corresponds to a
        trial number and eye number (0 is trial 0, eye 0, 1 is trial 0, eye 1,
        2 is trial 1, eye 0, etc.). Each array is the time difference between
        adjacent samples for that trial and eye.

    Returns
    -------
    dict[str, np.float64]
    A dictionary of statistics of the data, including the mean, min, 5th 
    percentile, median, 95th percentile, 99th percentile, and maximum values.
    """
    t_deltas = np.concatenate(t_deltas)
    percentiles = [5, 50, 95, 99]
    dt_percentiles = np.percentile(t_deltas, percentiles)
    dt_stats = {
        "mean": np.mean(t_deltas),
        "min": dt_percentiles.min(),
        "5th percentile": dt_percentiles[0],
        "median": dt_percentiles[1],
        "95th percentile": dt_percentiles[2],
        "99th percentile": dt_percentiles[3],
        "max": dt_percentiles.max()
    }
    return dt_stats


def resample_data(participant_data: RawParticipantDataType, dt: float):
    for trial_data in participant_data:
        t_start = min([trial_data["data"][eye]["timestamp"].min() for eye in (0,1)])
        t_end = min([trial_data["data"][eye]["timestamp"].max() for eye in (0,1)])
        t_stop = round((t_end - t_start)/dt)*dt
        t_array = np.arange(stop=t_stop, step=dt)
        for eye_data in trial_data["data"]:
            t_old = eye_data["timestamp"] - t_start
            for t1, t2 in zip(t_array[:-1], t_array[1:]):
                index = np.where(t_old >= t1 and t_old < t2)
                


if __name__=="__main__":
    test_array = np.array([1, 2, 4, -1, 7])
    delta_array = calc_deltas(test_array)
    assert np.all(delta_array == np.array([1, 2, -5, 8]))