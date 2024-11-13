from numpy import typing as npt
from numpy.lib import recfunctions as rfn
import numpy as np
from .aliases import RawParticipantDataType, TrialDataType, ResampledParticipantDataType, pupil_datatype


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


def resample_data(participant_data: RawParticipantDataType, dt: float) -> ResampledParticipantDataType:
    """Resample all datasets for a list of participant data"""
    resampled_data = []
    for trial_data in participant_data:
        old_t_arrays = [trial_data["data"][eye]["timestamp"] for eye in (0, 1)]
        t_start, t_stop, t_ins = get_key_times(old_t_arrays, trial_data["attributes"]["t_instruction"], dt)
        resampled_array = resample_trial(trial_data, t_start, t_stop, dt)
        keys = ("die", "recording", "task", "trial")
        resampled_attributes = {k: trial_data["attributes"][k] for k in keys}
        resampled_attributes.update({"t_offset": t_start, "t_instruction": t_ins, "sample_time_interval": dt})
        resampled_data.append(
            {"attributes": resampled_attributes, "data": resampled_array.copy()}
        )
    return resampled_data


def resample_trial(trial_data: TrialDataType, t_start: float, t_stop: float, dt: float) -> np.ndarray:
    """Resample the data for both eyes for a single trial

    Returns
    -------
    resampled_array: np.ndarray
        A structured NumPy array where each row corresponds to a single data sample, and
        each column corresponds to one of the eyes. For example, resampled_array[0,1] is
        the first data sample (point) for eye 1, and resampled_array[:,0] is all data 
        for eye 0.
    """
    t_array: npt.NDArray[np.float64] = np.arange(stop=t_stop, step=dt, dtype=np.float64)
    resampled_array = np.zeros((t_array.size, 2), dtype=trial_data["data"][0].dtype)
    for n_eye, eye_data in enumerate(trial_data["data"]):
        t_old = eye_data["timestamp"] - t_start
        weight_matrix, conf_array = calc_weight_and_confidence(
            eye_data, t_array, t_old, dt
        )
        resampled_array[:, n_eye] = resample_dataset(
            eye_data, weight_matrix, t_array, conf_array
        )
    return resampled_array


def get_key_times(time_arrays: list[npt.NDArray[np.float64]], t_instruction: float, dt: float) -> tuple[float, float, float]:
    t_start = min([time_arrays[eye][0] for eye in (0, 1)])
    t_end = min([time_arrays[eye][-1] for eye in (0, 1)])
    t_stop = round((t_end - t_start) / dt) * dt
    t_ins = round((t_instruction - t_start)/dt)*dt
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
    for i, t in enumerate(t_array):
        if t < dt:
            # The first time value does not have any data before it, so use
            # the initial condition
            index = np.full_like(t_old, False, dtype=np.bool)
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
        data: np.ndarray = data_group["data"]
        max_length = max(max_length, data.shape[0])
    return max_length


def convert_to_array(participant_data: list[dict[str, dict | np.ndarray]]) -> dict[str, np.ndarray]:
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
    input_dtype = participant_data[0]["data"].dtype
    array_dict = {
        "action": np.full((N_max, N_task["action"], 2), fill_value=np.nan, dtype=input_dtype),
        "observation": np.full((N_max, N_task["observation"], 2), fill_value=np.nan, dtype=input_dtype),
    }
    i_tasks = {"action": 0, "observation": 0}
    for data_group in participant_data:
        task = data_group["attributes"]["task"]
        trial_length = data_group["data"].shape[0]
        i = i_tasks[task]
        for n_eye in (0, 1):
            array_dict[task][0:trial_length, i, n_eye] = data_group["data"][:,n_eye]
        i_tasks[task] += 1
    return array_dict


def get_trendlines_by_task(data_array):
    N_max = data_array.shape[0]
    trendline_array = np.full((N_max, 2, 2, 3), fill_value=np.nan, dtype=np.float64)
    trendline_array[:,:,:,0] = np.nanpercentile(data_array, (5), axis=1)
    trendline_array[:,:,:,1] = np.mean(data_array, axis=1, where=~np.isnan(data_array))
    trendline_array[:,:,:,2] = np.nanpercentile(data_array, (95), axis=1)
    return trendline_array


def normalize_pupil_diameter(pupil_data: np.ndarray, t_baseline: float = 1.0):
    """Normalize the input data to a mean of 0 for the first t_baseline seconds"""
    t = pupil_data["timestamp"][:,0,0]
    i_baseline = np.max(np.nonzero(t < t_baseline))
    d = pupil_data["diameter_3d"]
    d_mean = np.nanmean(d[:i_baseline,:,:], axis=0)
    if np.any(np.isnan(d_mean)):
        # Sometimes the entire baseline has low confidence, resulting in a NaN mean.
        # Replace the NaN means by the mean of all baselines across all trials for each
        # eye.
        d_mean_by_eye = np.nanmean(d[:i_baseline,:,:], axis=(0, 1))
        d_mean = np.where(np.isnan(d_mean), d_mean_by_eye, d_mean)
    pupil_data["diameter_3d"] = d / d_mean - 1.0


def remove_low_confidence(data_array: np.ndarray, confidence_threshold: float = 0.6):
    """Replace low-confidence data in dataset with NaN

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
    data_fields = get_other_fields(("timestamp", "confidence", "world_index"), data_array.dtype)
    confidence = data_array["confidence"]
    low_conf_index = np.where(confidence < confidence_threshold)
    data_array[data_fields][low_conf_index] = np.nan

def get_max_values(data_array: np.ndarray) -> np.ndarray:
    return np.nanmax(data_array, axis=0)


def calc_split(class_distribution: np.ndarray, bin_edges: np.ndarray) -> tuple[float, tuple[float, float]]:
    # Find pupil diameter increase that maximally separates the classes
    # Brute force SVM solution based on squared hinge loss:
    min_loss = np.inf
    zeros = np.zeros_like(bin_edges[1:])
    for b in bin_edges[1:]:
        y = 2*(bin_edges[0:-1] - b)/bin_edges[1] + 1
        loss = class_distribution[:, 0]*(np.maximum(zeros, 1 - y)**2)
        loss += class_distribution[:, 1]*(np.maximum(zeros, 1 + y)**2)
        loss = loss.sum()
        if loss < min_loss:
            min_loss = loss
            split = b
        else:
            break
    return split


def count_tasks(participant_data: list[dict[str, dict | np.ndarray]]) -> dict[str, int]:
    """Count the number of times each task type appears in the dataset.
    
    Although the number of tasks is balanced at 60 each for most datasets, there are 
    rare exceptions where a trial was restarted and a perfect balance is not guaranteed.
    """
    n = {"action": 0, "observation": 0}
    for trial_data in participant_data:
        task = trial_data["attributes"]["task"]
        n[task] += 1
    return n


def interpolate_nan(data_array: np.ndarray):
    """Linearly interpolate data where it is NaN
    
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


def get_other_fields(fields: list, dtype: np.dtype) -> list:
    """Get all the fields in a numpy structured array other than the ones given"""
    return [name for name in dtype.names if name not in fields]


if __name__ == "__main__":
    test_array = np.array([1, 2, 4, -1, 7])
    delta_array = calc_deltas(test_array)
    assert np.all(delta_array == np.array([1, 2, -5, 8]))
