import numpy as np
import collections.abc as abc


planar_position_dt = np.dtype([("x", np.double), ("y", np.double)])
position_dt = np.dtype([("x", np.double), ("y", np.double), ("z", np.double)])
axes_dt = np.dtype([("a", np.double), ("b", np.double)])
ellipse_dt = np.dtype(
    [("center", planar_position_dt), ("axes", axes_dt), ("angle", np.double)]
)
sphere_dt = np.dtype([("center", position_dt), ("radius", np.double)])
circle_3d_dt = np.dtype(
    [("center", position_dt), ("normal", position_dt), ("radius", np.double)]
)
projected_sphere_dt = np.dtype(
    [("center", planar_position_dt), ("axes", axes_dt), ("angle", np.double)]
)


pupil_datatype = np.dtype(
    [
        ("timestamp", np.double),
        ("world_index", np.longlong),
        ("confidence", np.double),
        ("norm_pos", planar_position_dt),
        ("diameter", np.double),
        ("ellipse", ellipse_dt),
        ("diameter_3d", np.double),
        ("sphere", sphere_dt),
        ("circle_3d", circle_3d_dt),
        ("theta", np.double),
        ("phi", np.double),
        ("projected_sphere", projected_sphere_dt),
    ]
)


pupil_to_csv_fieldmap = {
    "timestamp": "pupil_timestamp",
    "world_index": "world_index",
    "id": "eye_id",
    "confidence": "confidence",
    "norm_pos": ["norm_pos_x", "norm_pos_y"],
    "diameter": "diameter",
    "method": "method",
    "ellipse": {
        "center": ["ellipse_center_x", "ellipse_center_y"],
        "axes": ["ellipse_axis_a", "ellipse_axis_b"],
        "angle": "ellipse_angle",
    },
    "diameter_3d": "diameter_3d",
    "model_confidence": "model_confidence",
    "sphere": {
        "center": ["sphere_center_x", "sphere_center_y", "sphere_center_z"],
        "radius": "sphere_radius",
    },
    "circle_3d": {
        "center": [
            "circle_3d_center_x",
            "circle_3d_center_y",
            "circle_3d_center_z",
        ],
        "normal": [
            "circle_3d_normal_x",
            "circle_3d_normal_y",
            "circle_3d_normal_z",
        ],
        "radius": "circle_3d_radius",
    },
    "theta": "theta",
    "phi": "phi",
    "projected_sphere": {
        "center": [
            "projected_sphere_center_x",
            "projected_sphere_center_y",
        ],
        "axes": [
            "projected_sphere_axis_a",
            "projected_sphere_axis_b",
        ],
        "angle": "projected_sphere_angle",
    },
}


def get_flattened_values(input_dict: dict) -> tuple[str]:
    """Get a tuple of values in a nested dictionary of strings and lists of strings
    
    The purpose of this function is to generate a list of field names from the mapping
    between the names of the keys in the nested dictionaries stored in the pldata files
    and the columns of the CSV file exported by Pupil Player.

    Parameters
    ----------
    input_dict : dict
        A nested dictionary of dictionaries, where the value at the end of each "branch"
        is either a string or a list of strings.

    Returns
    -------
    tuple[str]
        All "leaves" at the end of all of the branches of the nested dictionary.

    Example
    -------
    >>> nested_dict = {
    ...     "timestamp": "pupil_timestamp",
    ...     "norm_pos": ["norm_pos_x", "norm_pos_y"],
    ...     "ellipse": {
    ...         "center": ["ellipse_center_x", "ellipse_center_y"],
    ...         "axes": ["ellipse_axis_a", "ellipse_axis_b"],
    ...         "angle": "ellipse_angle"
    ...     }
    ... }
    >>> get_flattened_values(nested_dict)
    ('pupil_timestamp', 'norm_pos_x', 'norm_pos_y', 'ellipse_center_x', 
    'ellipse_center_y', 'ellipse_axis_a', 'ellipse_axis_b', 'ellipse_angle')
    """
    flattened_values = []
    for value in input_dict.values():
        match value:
            case str():
                flattened_values.append(value)
            case list():
                flattened_values.extend(value)
            case dict():
                flattened_values.extend(get_flattened_values(value))
    return tuple(flattened_values)


def get_csv_fieldnames() -> tuple[str]:
    return get_flattened_values(pupil_to_csv_fieldmap)


class FieldMapKeyError(Exception):
    """Raised when a key is not included in a field map"""

    pass


if __name__=='__main__':
    nested_dict = {
        "timestamp": "pupil_timestamp",
        "world_index": "world_index",
        "id": "eye_id",
        "confidence": "confidence",
        "norm_pos": ["norm_pos_x", "norm_pos_y"],
        "diameter": "diameter",
        "method": "method",
        "ellipse": {
            "center": ["ellipse_center_x", "ellipse_center_y"],
            "axes": ["ellipse_axis_a", "ellipse_axis_b"],
            "angle": "ellipse_angle",
        },
    }
    print(get_flattened_values(nested_dict))