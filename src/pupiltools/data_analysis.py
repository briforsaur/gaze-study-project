from dataclasses import dataclass
from numpy import typing as npt
from numpy.lib import recfunctions as rfn
import numpy as np
import numpy.typing as npt
import scipy.signal as sig
from sklearn.impute import SimpleImputer
import logging
from collections.abc import Iterable
from .aliases import (
    RawParticipantDataType,
    TrialDataType,
    ResampledParticipantDataType,
    pupil_datatype,
)


logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    ftype: str
    btype: str
    N: int
    Wn: float
    fs: float


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
        for eye in (0, 1):
            t = trial["data"][eye]["timestamp"]  # type: ignore
            assert isinstance(t, np.ndarray)
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
    t_delta_array = np.concatenate(t_deltas)
    percentiles = [5, 50, 95, 99]
    dt_percentiles = np.percentile(t_delta_array, percentiles)
    dt_stats = {
        "mean": np.mean(t_delta_array),
        "min": t_delta_array.min(),
        "5th percentile": dt_percentiles[0],
        "median": dt_percentiles[1],
        "95th percentile": dt_percentiles[2],
        "99th percentile": dt_percentiles[3],
        "max": t_delta_array.max(),
    }
    return dt_stats


def resample_data(
    participant_data: RawParticipantDataType, dt: float
) -> ResampledParticipantDataType:
    """Resample all datasets for a list of participant data"""
    resampled_data = []
    for trial_data in participant_data:
        old_t_arrays = [trial_data["data"][eye]["timestamp"] for eye in (0, 1)]  # type: ignore
        t_start, t_stop, t_ins = get_key_times(
            old_t_arrays, trial_data["attributes"]["t_instruction"], dt  # type: ignore
        )
        resampled_array = resample_trial(trial_data, t_start, t_stop, dt)
        keys = ("die", "recording", "task", "trial")
        resampled_attributes = {k: trial_data["attributes"][k] for k in keys}  # type: ignore
        resampled_attributes.update(
            {"t_offset": t_start, "t_instruction": t_ins, "sample_time_interval": dt}
        )
        resampled_data.append(
            {"attributes": resampled_attributes, "data": resampled_array.copy()}
        )
    return resampled_data


def resample_trial(
    trial_data: TrialDataType, t_start: float, t_stop: float, dt: float
) -> np.ndarray:
    """Resample the data for both eyes for a single trial

    Returns
    -------
    resampled_array: np.ndarray
        A structured NumPy array where each row corresponds to a single data sample, and
        each column corresponds to one of the eyes. For example, resampled_array[0,1] is
        the first data sample (point) for eye 1, and resampled_array[:,0] is all data
        for eye 0.
    """
    t_array: npt.NDArray[np.float64] = np.arange(
        start=0, stop=t_stop, step=dt, dtype=np.float64
    )
    resampled_array = np.zeros((t_array.size, 2), dtype=trial_data["data"][0].dtype)  # type: ignore
    for n_eye, eye_data in enumerate(trial_data["data"]):
        assert isinstance(eye_data, np.ndarray)
        t_old = eye_data["timestamp"] - t_start
        weight_matrix, conf_array = calc_weight_and_confidence(
            eye_data, t_array, t_old, dt
        )
        resampled_array[:, n_eye] = resample_dataset(
            eye_data, weight_matrix, t_array, conf_array
        )
    return resampled_array


def get_key_times(
    time_arrays: list[npt.NDArray[np.float64]], t_instruction: float, dt: float
) -> tuple[float, float, float]:
    t_start = min([time_arrays[eye][0] for eye in (0, 1)])
    t_end = min([time_arrays[eye][-1] for eye in (0, 1)])
    t_stop = round((t_end - t_start) / dt) * dt
    t_ins = round((t_instruction - t_start) / dt) * dt
    return t_start, t_stop, t_ins


def calc_weight_and_confidence(
    eye_data: np.ndarray,
    t_array: npt.NDArray[np.float64],
    t_old: npt.NDArray[np.float64],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the weight matrix for weighted average resampling and confidence"""
    # Computing weights for a weighted average based on confidence
    weight_matrix = np.zeros((t_array.size, t_old.size), dtype=np.float64)
    conf_array = np.zeros_like(t_array)
    # Preallocate index array
    index = np.full_like(t_old, False, dtype=np.bool)
    for i, t in enumerate(t_array):
        if t < dt:
            # The first time value does not have any data before it, so use
            # the initial condition
            index[0] = True
        else:
            elements_in_time_interval = np.logical_and(t_old >= t - dt, t_old < t)
            # Check if any values were found in the time interval, falling back
            # to the previous index value if not (effectively a zero-order hold)
            if np.any(elements_in_time_interval):
                index = elements_in_time_interval
        weight_row = eye_data["confidence"] * index
        # The confidence values should be averaged, not weighted averaged
        conf_array[i] = weight_row.mean(where=index)
        weight_sum = weight_row.sum()
        # Checking the sum of weights is greater than 0 before division. The sum
        # could be zero if the confidence values are all zero.
        if weight_sum > 0.001:
            weight_matrix[i, :] = weight_row / weight_sum
        else:
            weight_matrix[i, :] = index
    return weight_matrix, conf_array


def resample_dataset(
    eye_data: np.ndarray,
    weight_matrix: npt.NDArray[np.float64],
    t_array: npt.NDArray[np.float64],
    conf_array: npt.NDArray[np.float64],
):
    """Resample a single structured eye data array via weighted average"""
    # Must convert the structured array to an unstructured array for matrix
    # multiplication, then convert back to structured
    unstructured_data = rfn.structured_to_unstructured(eye_data)
    unstruct_resampled_data = weight_matrix @ unstructured_data
    resampled_array = rfn.unstructured_to_structured(
        unstruct_resampled_data, dtype=eye_data.dtype
    )
    # Replace weighted average time & confidence with correct time & confidence
    resampled_array["timestamp"] = t_array
    resampled_array["confidence"] = conf_array
    if "world_index" in eye_data.dtype.names:
        # Need to correct world index: the result of the matrix multiplication between
        # an int (world_index) and float (weight_matrix) casts the result as a float.
        # When the unstructured array is converted back to a structured array, the
        # floating-point multiplication result is cast to an int, effectively applying
        # a floor rather than a round function.
        world_index_array = weight_matrix @ eye_data["world_index"]
        resampled_array["world_index"] = np.rint(world_index_array)
    return resampled_array


def get_max_data_length(data_list: ResampledParticipantDataType) -> int:
    max_length = 0
    for data_group in data_list:
        data: np.ndarray = data_group["data"]  # type: ignore
        max_length = max(max_length, data.shape[0])
    return max_length


def convert_to_array(
    participant_data: ResampledParticipantDataType,
) -> dict[str, np.ndarray]:
    """Converts a list of task metadata and data to a single numpy array

    Returns
    -------
    dict[str, numpy.ndarray]
        A dictionary of structured NxN_tasksx2 array, where the first index is the
        sample number up to N (the maximum length of any of the data series), the
        second index is the task number, in order, for a given task (usually 60, with
        a couple of exceptions), the third index is the eye ID (0 for right,
        1 for left), and the fourth index is the task type (0 for action, 1 for
        observation). For example, output[:, 0, 0, 1] is all samples for the first
        observation trial, for the right eye. All trial data series that are shorter
        than the longest series are padded with np.nan.
    """
    N_max = get_max_data_length(participant_data)
    N_task = count_tasks(participant_data)
    input_dtype = participant_data[0]["data"].dtype  # type: ignore
    array_dict = {
        "action": np.full(
            (N_max, N_task["action"], 2), fill_value=np.nan, dtype=input_dtype
        ),
        "observation": np.full(
            (N_max, N_task["observation"], 2), fill_value=np.nan, dtype=input_dtype
        ),
    }
    i_tasks = {"action": 0, "observation": 0}
    for data_group in participant_data:
        task = data_group["attributes"]["task"]
        assert isinstance(task, str)
        trial_length = data_group["data"].shape[0]  # type: ignore
        i = i_tasks[task]
        for n_eye in (0, 1):
            array_dict[task][0:trial_length, i, n_eye] = data_group["data"][:, n_eye]  # type: ignore
        i_tasks[task] += 1
    return array_dict


def get_trendlines_by_task(data_array: np.ndarray):
    """Get the mean, and 5th and 95th percentile of an array of task data

    Parameters
    ----------
    data_array: numpy.ndarray
        An N x N_task x ... array, where N is the number of samples in time and N_task
        is the number of timeseries (number of tasks).

    Returns
    -------
    trendline_array: numpy.ndarray
        An N x ... x 3 array, where the first dimension is time, and the final dimension
        is the three separate statistics calculated across all tasks for each instant in
        time. The first entry in the final column is the 5th percentile, the second is
        the mean, and the third is the 95th percentile.
    """
    trendline_shape = (*np.squeeze(data_array[:, 0, ...]).shape, 3)
    trendline_array = np.full(trendline_shape, fill_value=np.nan, dtype=np.float64)
    trendline_array[..., 0] = np.nanpercentile(data_array, (5), axis=1)
    trendline_array[..., 1] = np.mean(data_array, axis=1, where=~np.isnan(data_array))
    trendline_array[..., 2] = np.nanpercentile(data_array, (95), axis=1)
    return trendline_array


def normalize_pupil_diameter(pupil_data: np.ndarray, t_baseline: float = 1.0):
    """Normalize the input data to a mean of 0 for the first t_baseline seconds"""
    t = pupil_data["timestamp"][:, 0, 0]
    i_baseline = np.max(np.nonzero(t < t_baseline))
    d = pupil_data["diameter_3d"]
    d_mean = np.nanmean(d[:i_baseline, :, :], axis=0)
    if np.any(np.isnan(d_mean)):
        # Sometimes the entire baseline has low confidence, resulting in a NaN mean.
        # Replace the NaN means by the mean of all baselines across all trials for each
        # eye.
        d_mean_by_eye = np.nanmean(d[:i_baseline, :, :], axis=(0, 1))
        d_mean = np.where(np.isnan(d_mean), d_mean_by_eye, d_mean)
    pupil_data["diameter_3d"] = d / d_mean - 1.0


def remove_low_confidence(data_array: np.ndarray, confidence_threshold: float = 0.6):
    """Replace low-confidence data in dataset with NaN

    Modifies the input array in-place.

    Parameters
    ----------
    data_array: numpy.ndarray
        A structured array with fields based on the pupil datatype. This array is
        modified in-place. All fields except for "timestamp", "confidence", and
        "world_index" will be affected.
    confidence_threshold: float = 0.6
        The confidence value below which data is replaced by NaN. Must be between 0 and
        1.
    """
    data_fields = get_other_fields(
        ("timestamp", "confidence", "world_index"), data_array.dtype
    )
    confidence = data_array["confidence"]
    low_conf_index = np.nonzero(confidence < confidence_threshold)
    data_array[data_fields][low_conf_index] = np.nan


def get_max_values(data_array: np.ndarray) -> np.ndarray:
    return np.nanmax(data_array, axis=0)


def calc_split(
    class_distribution: np.ndarray, bin_edges: np.ndarray
) -> tuple[float, tuple[float, float]]:
    # Find pupil diameter increase that maximally separates the classes
    # Brute force SVM solution based on squared hinge loss:
    min_loss = np.inf
    zeros = np.zeros_like(bin_edges[1:])
    for b in bin_edges[1:]:
        y = 2 * (bin_edges[0:-1] - b) / bin_edges[1] + 1
        loss = class_distribution[:, 0] * (np.maximum(zeros, 1 - y) ** 2)
        loss += class_distribution[:, 1] * (np.maximum(zeros, 1 + y) ** 2)
        loss = loss.sum()
        if loss < min_loss:
            min_loss = loss
            split = b
        else:
            break
    return split  # type: ignore


def count_tasks(participant_data: ResampledParticipantDataType) -> dict[str, int]:
    """Count the number of times each task type appears in the dataset.

    Although the number of tasks is balanced at 60 each for most datasets, there are
    rare exceptions where a trial was restarted and a perfect balance is not guaranteed.
    """
    n = {"action": 0, "observation": 0}
    for trial_data in participant_data:
        task = trial_data["attributes"]["task"]
        assert isinstance(task, str)
        n[task] += 1
    return n


def interpolate_nan(data_array: np.ndarray):
    """Linearly interpolate data where it is NaN

    Modifies the input array in-place.

    All fields except for "timestamp" and "confidence" are checked for NaNs. NaNs
    surrounded by non-NaN data are filled in with linear interpolation. NaNs with no
    preceding non-NaN data (e.g. at the start of the array) are filled in with the first
    non-NaN value. NaNs with no following non-NaN data (e.g. at the end of the array)
    are replaced with the last non-NaN value, unless they extend past the defined end
    of the recording (where timestamps are NaN).
    """
    data_fields = get_other_fields(("timestamp", "confidence"), data_array.dtype)
    for data_field in data_fields:
        for i_trial in range(data_array.shape[1]):
            for i_eye in range(data_array.shape[2]):
                t = data_array["timestamp"][:, i_trial, i_eye]
                data = data_array[data_field][:, i_trial, i_eye]
                # Since recordings shorter than the longest recording are padded with
                # nans, we need to avoid interpolating after the end of the recording
                non_nan = ~np.isnan(t)
                t = t[non_nan]
                y: np.ndarray = data[non_nan]
                nans = np.isnan(y)
                if t[~nans].size != 0:
                    y[nans] = np.interp(t[nans], t[~nans], y[~nans])
                data[non_nan] = y


def get_other_fields(fields: Iterable[str], dtype: np.dtype) -> list:
    """Get all the fields in a numpy structured array other than the ones given"""
    if dtype.names is not None:
        other_fields = [name for name in dtype.names if name not in fields]
    else:
        other_fields = []
    return other_fields


def filter_signal(x: np.ndarray, filter_config: FilterConfig) -> np.ndarray:
    """Filter an array of timeseries signals.
    
    Filter an array of timeseries signals over the time dimension. The time
    dimension is assumed to be the first dimension in the array. For example,
    an array with 1000 samples in time, over sixty trials, for each eye, would
    have the shape (1000, 60, 2).

    Parameters
    ----------
    x: numpy.ndarray
        The array of signals to be filtered, of shape (n, ...) where n
        is the number of samples in time.
    filter_config: FilterConfig
        An object with properties corresponding to the order (N), cutoff 
        frequency (Wn), band type (btype), filter type (ftype), and
        sample rate (fs) of a scipy.signal.iirfilter second-order-sections
        filter.
    
    Returns
    -------
    x_filt: np.ndarray
        The array of filtered signals, of shape (n, ...).
    """
    sos = sig.iirfilter(
        N=filter_config.N,
        Wn=filter_config.Wn,
        btype=filter_config.btype,
        ftype=filter_config.ftype,
        output="sos",
        fs=filter_config.fs,
    )
    # zi required to match initial conditions
    zi = sig.sosfilt_zi(sos)
    # Reshaping zi, ex. if x has shape (N, 60, 2), then zi must be (n_sections, 2, 1, 1)
    extra_dims = [1]*(np.ndim(x) - 1)
    zi = np.reshape(zi, (2, 2, *extra_dims))
    # Getting the initial condition of x and reshaping, for example if x has shape
    # (N, 60, 2), the final shape must be (1, 1, 60, 2) to broadcast with zi
    x0 = np.reshape(x[0,:,:], (1, 1, *x.shape[1:]))
    x_filt = sig.sosfilt(sos, x, axis=0, zi=zi*x0)
    # Only return the first element of x_filt, the second gives delay from zi
    return x_filt[0]  # type: ignore


def calc_rate_of_change(data: np.ndarray, dt: float = 0.01):
    v_data = np.full_like(data, fill_value=0.0)
    v_data[1:-1] = (data[1:-1] - data[0:-2]) / dt
    return v_data


def get_features(
    data: np.ndarray, dt: float = 0.01, t_range: tuple[float, float] = (0.0, np.inf)
) -> np.ndarray:
    # Calculate the index range based on the time range (assumes time starts at 0)
    i_range = np.array(
        [
            np.floor(t_range[0] / dt),
            np.min([np.floor(t_range[1] / dt) + 1, data.shape[0]]),
        ]
    )
    i_range = i_range.astype(np.int64)
    v_data = calc_rate_of_change(data, dt)
    feature_list = (
        np.nanmean(data[i_range[0] : i_range[1], ...], axis=0),
        np.nanmax(data[i_range[0] : i_range[1], ...], axis=0),
        np.nanmean(np.abs(v_data[i_range[0] : i_range[1], ...]), axis=0),
        np.nanmax(v_data[i_range[0] : i_range[1], ...], axis=0),
    )
    features = np.concat(feature_list, axis=1)
    return features


def get_timeseries(data: np.ndarray, i_range: tuple[int, int]) -> np.ndarray:
    """Reshape timeseries data in a form amenable to neural net input
    
    Parameters
    ----------
    data: np.ndarray
        The data to be reshaped, assumed to be in shape N x n x 2, where N is the length
        in time, n is the number of tasks, and 2 corresponds to the number of eyes.
    i_range: tuple[int, int]
        The range of indices in time to return in the output vector. For example, for
        a timeseries with a sample rate of 0.01 seconds and starting time of 0, 
        i_range = [100, 201] would correspond to the all data points from 1.00 second to 
        2.00 seconds, inclusive.
    Returns
    -------
    reshaped_timeseries: np.ndarray
        The reshaped and pruned data in shape (n - p) x 2N. The data from each eye are
        concatenated along the columns. p is the number of tasks that were pruned from
        the final vector for having any NaNs in them. NaNs would be the result of 
        insufficiently long timeseries given the i_range, or extremely poor quality
        data that could not be interpolated.

    The number of pruned tasks, p, is printed to the logger.info stream.
    """
    timeseries = data[i_range[0] : i_range[1], ...]
    timeseries = np.concat(timeseries, axis=1)
    # Remove any rows with NaN
    timeseries_pruned = timeseries[~np.isnan(timeseries).any(axis=1)]
    if timeseries_pruned.shape[0] < timeseries.shape[0]:
        N = timeseries.shape[0] - timeseries_pruned.shape[0]
        logger.info(f"Removed {N} samples due to NaNs.")
    return timeseries_pruned


def calc_index_from_time(N: int, t_range: tuple[float, float], dt: float = 0.01) -> tuple[int, int]:
    """Calculate an index range based on a time range and sample rate

    Parameters
    ----------
    N: int
        The number of samples in time in the array.
    t_range: tuple[float, float]
        The time range to be converted into an index range. Assumes t = 0 at index 0.
    dt: float = 0.01
        The sample rate of the data in the array. Must be in the same unit as t_range.
    Returns
    -------
    i_range: ndarray[int]
        The pair of indices closest to the given time range values, rounded down. Will
        not exceed N.
    Example
    -------
    >>> calc_index_from_time(51, (0.1, 0.4))
    (10, 41)
    >>> calc_index_from_time(51, (0.1, 0.6))
    (10, 51)
    >>> calc_index_from_time(101, (1.0, 3.0), dt=0.1)
    (10, 31)
    """
    # Calculate the index range based on the time range (assumes time starts at 0)
    i_range = np.array(
        [
            np.floor(t_range[0] / dt),
            np.min([np.floor(t_range[1] / dt) + 1, N]),
        ]
    )
    i_range = i_range.astype(int)
    return tuple(i_range)


def impute_missing_values(feature_array: np.ndarray) -> np.ndarray:
    total_nan = np.sum(np.isnan(feature_array))
    logger.info(f"Imputing {total_nan} missing values.")
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    feature_array = mean_imputer.fit_transform(feature_array)
    return feature_array


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_array = np.array([1, 2, 4, -1, 7])
    delta_array = calc_deltas(test_array)
    assert np.all(delta_array == np.array([1, 2, -5, 8]))
