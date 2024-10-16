from numpy import typing as npt
import numpy as np
from .aliases import RawParticipantDataType, TrialDataType


def calc_deltas(array: np.ndarray) -> np.ndarray:
    """Calculates the differences between adjacent values in an array
    
    For an input array of length N, the function returns an N-1 array
    of the difference between elements n+1 and n, from n = 0 to N-2"""
    return array[1:] - array[:-1]


def get_time_deltas(participant_data: RawParticipantDataType) -> list[np.ndarray]:
    t_deltas = []
    trial: TrialDataType
    for trial in participant_data:
        for eye in (0,1):
            t = trial["data"][eye]["timestamp"]
            dt = calc_deltas(t)
            t_deltas.append(dt)
    return t_deltas


def get_time_delta_stats(t_deltas: list[np.ndarray]) -> npt.NDArray[np.float64]:
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


if __name__=="__main__":
    test_array = np.array([1, 2, 4, -1, 7])
    delta_array = calc_deltas(test_array)
    assert np.all(delta_array == np.array([1, 2, -5, 8]))