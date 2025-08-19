# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

import numpy as np
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
        location: list[float],  # type: ignore
        model_confidence: float,
        world_index: int = -1,
    ) -> None:
        self.timestamp = timestamp
        self.world_index = world_index
        self.confidence = confidence
        self.norm_pos = Cartesian2D(*norm_pos)
        self.diameter = diameter
        self.ellipse = Ellipse(**ellipse)  # type: ignore
        self.diameter_3d = diameter_3d
        self.sphere = Sphere(**sphere)  # type: ignore
        self.circle_3d = Circle3D(**circle_3d)  # type: ignore
        self.theta = theta
        self.phi = phi
        self.projected_sphere = ProjectedSphere(**projected_sphere)  # type: ignore
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
                case (
                    Cartesian2D()
                    | Cartesian3D()
                    | Ellipse()
                    | Sphere()
                    | Circle3D()
                    | ProjectedSphere()
                ):
                    data_list.append(astuple(field_value))
        return tuple(data_list)
    
    def fields_to_tuple_for_csv(self) -> tuple:
        field_values = list(self.fields_to_tuple())
        field_values.append(self.id)
        return tuple(field_values)


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

    def __init__(
        self,
        timestamp: float,
        confidence: float,
        norm_pos: list[float],
        base_data: list[dict[str, Any]],
        gaze_point_3d: list[float],
        eye_centers_3d: dict[str, list[float]] | None = None,
        gaze_normals_3d: dict[str, list[float]] | None = None,
        topic: str = "",
        world_index: int = -1,
        **kw,
    ) -> None:
        self.timestamp = timestamp
        self.world_index = world_index
        self.confidence = confidence
        self.norm_pos = Cartesian2D(*norm_pos)
        self.base_data = [PupilData(**data) for data in base_data]
        self.gaze_point_3d = Cartesian3D(*gaze_point_3d)
        if eye_centers_3d is not None and gaze_normals_3d is not None:
            self.eye_centers_3d = [
                Cartesian3D(*eye_center) for eye_center in eye_centers_3d.values()
            ]
            self.gaze_normals_3d = [
                Cartesian3D(*gaze_normal) for gaze_normal in gaze_normals_3d.values()
            ]
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
    
    def fields_to_tuple_for_csv(self) -> tuple:
        return self.fields_to_tuple()

    def to_numpy_recarray(self) -> np.ndarray:
        data = self.fields_to_tuple()
        return np.array(data, dtype=gaze_datatype)
    

def _get_csv_header(class_obj: type) -> list[str]:
    header_list = []
    for field in fields(class_obj):
        f_type = field.type
        f_name = field.name
        if f_type == float or f_type == int or f_type == str:
            header_list.append(f_name)
        elif f_type == list[PupilData]:
            field_names = [f"{f_name}_{i}_timestamp" for i in range(2)]
            header_list.extend(field_names)
        elif f_type == list[Cartesian3D]:
            sub_names = _get_csv_header(Cartesian3D)
            for i in range(2):
                field_names = [f"{f_name}_{i}_{sub_name}" for sub_name in sub_names]
                header_list.extend(field_names)
        else:
            sub_names = _get_csv_header(f_type) # type: ignore
            field_names = [f"{f_name}_{sub_name}" for sub_name in sub_names]
            header_list.extend(field_names)
    if class_obj == PupilData:
        header_list.append("eye_id")
    return header_list


PUPIL_CSV_FIELDS = _get_csv_header(PupilData)
GAZE_CSV_FIELDS = _get_csv_header(GazeData)


def flatten_nested_tuple(data: tuple[Any]) -> tuple[Any]:
    data_list = []
    for item in data:
        match item:
            case float() | int() | str():
                data_list.append(item)
            case tuple():
                sub_items = flatten_nested_tuple(item)
                data_list.extend(sub_items)
    return tuple(data_list)


if __name__ == "__main__":
    print(_get_csv_header(PupilData))
    print(_get_csv_header(GazeData))
    test_p_dict = {
        'id': 0, 
        'topic': 'pupil.0.3d', 
        'method': 'pye3d 0.3.0 real-time', 
        'norm_pos': [0.24351970741892207, 0.44267443593140643], 
        'diameter': 27.32890140708419, 
        'confidence': 1.0, 
        'timestamp': 345.96117200001027, 
        'sphere': {
            'center': [0.26086557200170474, -2.328750597080022, 39.13150616664784], 
            'radius': 10.392304845413264
        }, 
        'projected_sphere': {
            'center': [97.74302257519636, 79.10477264712452], 
            'axes': [180.19507618312036, 180.19507618312036], 
            'angle': 0.0
        }, 
        'circle_3d': {
            'center': [-4.80148189549971, 1.1571421916810967, 30.751697721131602], 
            'normal': [-0.48712461218222697, 0.3354301899929012, -0.806347443629383], 
            'radius': 1.2889188296557954
        }, 
        'diameter_3d': 2.577837659311591, 
        'ellipse': {
            'center': [46.75578382443304, 107.00650830116997], 
            'axes': [21.31778407928547, 27.32890140708419], 
            'angle': 148.42380933109075
        }, 
        'location': [46.75578382443304, 107.00650830116997], 
        'model_confidence': 1.0, 
        'theta': 1.228734488061224, 
        'phi': -2.1142342768113487
    }
    test_g_dict = {
        'eye_centers_3d': {
            '0': [19.98571631324934, 14.985869912781883, -19.92116119749855], 
            '1': [-40.022254973480145, 15.001520605051482, -19.949131602653416]
        }, 
        'gaze_normals_3d': {
            '0': [-0.10548705527122566, -0.39875290894360527, 0.9109712392711519], 
            '1': [0.07872946880333444, -0.41731550273288076, 0.9053449298034135]
        },
        'gaze_point_3d': [-14.411410769452052, -117.89830813333006, 275.84140173241457], 
        'norm_pos': [0.46460641707690487, 0.8831982210630259], 
        'topic': 'gaze.3d.01.', 
        'confidence': 1.0, 
        'timestamp': 345.9355710000091, 
        'base_data': [
            {'id': 0, 'topic': 'pupil.0.3d', 'method': 'pye3d 0.3.0 real-time', 'norm_pos': [0.24404183349315742, 0.4424241919490489], 'diameter': 27.47196090665017, 'confidence': 1.0, 'timestamp': 345.9354560000065, 'sphere': {'center': [0.26086557200170474, -2.328750597080022, 39.13150616664784], 'radius': 10.392304845413264}, 'projected_sphere': {'center': [97.74302257519636, 79.10477264712452], 'axes': [180.19507618312036, 180.19507618312036], 'angle': 0.0}, 'circle_3d': {'center': [-4.800008048072229, 1.19406680080027, 30.76626165241924], 'normal': [-0.48698279114739357, 0.3389832621620139, -0.8049460286878204], 'radius': 1.2953123337339543}, 'diameter_3d': 2.5906246674679085, 'ellipse': {'center': [46.85603203068622, 107.0545551457826], 'axes': [21.400981241810108, 27.47196090665017], 'angle': 148.15896622973645}, 'location': [46.85603203068622, 107.0545551457826], 'model_confidence': 1.0, 'theta': 1.224960364943982, 'phi': -2.114875498332212},
            {'id': 1, 'topic': 'pupil.1.3d', 'method': 'pye3d 0.3.0 real-time', 'norm_pos': [0.39628283763452093, 0.5210280400311091], 'diameter': 25.400197970234878, 'confidence': 1.0, 'timestamp': 345.93568600001163, 'sphere': {'center': [3.487830222643322, 2.4756545941550567, 42.027153211782256], 'radius': 10.392304845413264}, 'projected_sphere': {'center': [119.3480654960734, 112.59649163741423], 'axes': [167.46926491231036, 167.46926491231036], 'angle': 0.0}, 'circle_3d': {'center': [-1.4141897724228576, -0.9453525429509337, 33.526158556105706], 'normal': [-0.4716970939540645, -0.32918656525129575, -0.8180085921390711], 'radius': 1.3279301377834947}, 'diameter_3d': 2.6558602755669893, 'ellipse': {'center': [76.08630482582802, 91.96261631402706], 'axes': [21.109957465289888, 25.400197970234878], 'angle': 32.732852846330935}, 'location': [76.08630482582802, 91.96261631402706], 'model_confidence': 1.0, 'theta': 1.9062383248345953, 'phi': -2.0938628049026704}
        ]
    }
    test_p_data = PupilData(**test_p_dict)
    test_g_data = GazeData(**test_g_dict)
