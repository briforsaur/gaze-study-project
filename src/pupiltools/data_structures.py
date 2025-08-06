import numpy as np
import collections.abc as abc
from dataclasses import dataclass, fields, astuple
from typing import Any

from .aliases import pupil_datatype, gaze_datatype


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


@dataclass
class Cartesian2D:
    x: float
    y: float


@dataclass
class Cartesian3D(Cartesian2D):
    z: float


@dataclass
class Axes:
    a: float
    b: float


@dataclass(init=False)
class Ellipse:
    center: Cartesian2D
    axes: Axes
    angle: float

    def __init__(self, center: list[float], axes: list[float], angle: float) -> None:
        self.center = Cartesian2D(*center)
        self.axes = Axes(*axes)
        self.angle = angle


@dataclass(init=False)
class Sphere:
    center: Cartesian3D
    radius: float

    def __init__(self, center: list[float], radius: float) -> None:
        self.center = Cartesian3D(*center)
        self.radius = radius


@dataclass
class Circle3D:
    center: Cartesian3D
    normal: Cartesian3D
    radius: float

    def __init__(self, center: list[float], normal: list[float], radius: float) -> None:
        self.center = Cartesian3D(*center)
        self.normal = Cartesian3D(*normal)
        self.radius = radius


class ProjectedSphere(Ellipse):
    pass


@dataclass(init=False)
class PupilData:
    timestamp: float
    world_index: int
    confidence: float
    norm_pos: Cartesian2D
    diameter: float
    ellipse: Ellipse
    diameter_3d: float
    sphere: Sphere
    circle_3d: Circle3D
    theta: float
    phi: float
    projected_sphere: ProjectedSphere

    def __init__(
            self, 
            timestamp: float,
            confidence: float,
            norm_pos: list[float],
            diameter: float,
            ellipse: dict[str, list[float] | float],
            diameter_3d: float,
            sphere: dict[str, list[float] | float],
            circle_3d: dict[str, list[float] | float],
            theta: float,
            phi: float,
            projected_sphere: dict[str, list[float] | float],
            id: int,
            topic: str,
            method: str,
            location: list[float], #type: ignore
            model_confidence: float,
            world_index: int = -1) -> None:
        self.timestamp = timestamp
        self.world_index = world_index
        self.confidence = confidence
        self.norm_pos = Cartesian2D(*norm_pos)
        self.diameter = diameter
        self.ellipse = Ellipse(**ellipse) #type: ignore
        self.diameter_3d = diameter_3d
        self.sphere = Sphere(**sphere) #type: ignore
        self.circle_3d = Circle3D(**circle_3d) #type: ignore
        self.theta = theta
        self.phi = phi
        self.projected_sphere = ProjectedSphere(**projected_sphere) #type: ignore
        self.id = id
        self.topic = topic
        self.method = method
        self.location = location
        self.model_confidence = model_confidence
    
    def fields_to_tuple(self) -> tuple:
        """Create tuple only from fields, not all parameters"""
        data_list = []
        for field in fields(self):
            field_value = getattr(self, field.name)
            match field_value:
                case float() | int():
                    data_list.append(field_value)
                case Cartesian2D() | Cartesian3D() | Ellipse() | Sphere() | Circle3D() | ProjectedSphere():
                    data_list.append(astuple(field_value))
        return tuple(data_list)
    
    def to_numpy_recarray(self) -> np.ndarray:
        data = self.fields_to_tuple()
        return np.array(data, dtype=pupil_datatype)


@dataclass(init=False)
class GazeData:
    timestamp: float
    world_index: int
    confidence: float
    norm_pos: Cartesian2D
    base_data: list[PupilData]
    gaze_point_3d: Cartesian3D
    eye_centers_3d: list[Cartesian3D]
    gaze_normals_3d: list[Cartesian3D]
    topic: str

    def __init__(self, timestamp: float, confidence: float, norm_pos: list[float], base_data: list[dict[str, Any]], gaze_point_3d: list[float], topic: str, eye_centers_3d: dict[str, list[float]] = None, gaze_normals_3d: dict[str, list[float]] = None, world_index: int = -1, **kw) -> None:
        self.timestamp = timestamp
        self.world_index = world_index
        self.confidence = confidence
        self.norm_pos = Cartesian2D(*norm_pos)
        self.base_data = [PupilData(**data) for data in base_data]
        self.gaze_point_3d = Cartesian3D(*gaze_point_3d)
        if eye_centers_3d is not None:
            self.eye_centers_3d = [Cartesian3D(*eye_center) for eye_center in eye_centers_3d.values()]
            self.gaze_normals_3d = [Cartesian3D(*gaze_normal) for gaze_normal in gaze_normals_3d.values()]
        else:
            # Data for both eyes is not available
            # eye_centers_3d and gaze_normals_3d are not in the data message
            # The topic will be gaze.3d.0. or gaze.3d.1. instead of gaze.3d.01.
            eye_id = int(topic.split(".")[2])
            # eye_centers_3d and gaze_normals_3d are replaced by singular versions
            eye_center_3d = kw.pop("eye_center_3d")
            gaze_normal_3d = kw.pop("gaze_normal_3d")
            default = Cartesian3D(*np.full((3,), np.nan))
            self.eye_centers_3d = [default, default]
            self.gaze_normals_3d = [default, default]
            self.eye_centers_3d[eye_id] = Cartesian3D(*eye_center_3d)
            self.gaze_normals_3d[eye_id] = Cartesian3D(*gaze_normal_3d)
        self.topic = topic
        if world_index != -1:
            self.world_index = world_index

    def fields_to_tuple(self) -> tuple:
        data_list = []
        for field in fields(self):
            field_value = getattr(self, field.name)
            match field_value:
                case float() | int():
                    data_list.append(field_value)
                case Cartesian2D() | Cartesian3D():
                    data_list.append(astuple(field_value))
                case list() if isinstance(field_value[0], PupilData):
                    if len(field_value) > 1:
                        data = [values.timestamp for values in field_value]
                    else:
                        # One of the eyes was not available
                        data = [np.nan, np.nan]
                        eye_id = int(field_value[0].id)
                        data[eye_id] = field_value[0].timestamp
                    data_list.append(tuple(data))
                case list() if isinstance(field_value[0], Cartesian3D):
                    data = [astuple(values) for values in field_value]
                    data_list.append(tuple(data))
        return tuple(data_list)

    def to_numpy_recarray(self) -> np.ndarray:
        data = self.fields_to_tuple()
        return np.array(data, dtype=gaze_datatype)
    

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