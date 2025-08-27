# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

import numpy as np
from dataclasses import dataclass, fields, astuple
from typing import Any

from .aliases import pupil_datatype, gaze_datatype


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
    """A class for handling pupil data from PLDATA files

    After unpacking, PLDATA files contain lists of messages where each message is
    a deeply nested set of dictionaries (up to three layers). This format is difficult
    to work with and prevents IDE tools like tab completion for easy object 
    introspection. The PupilData class bundles this data into meaningful objects and
    allows easier conversion to useful datatypes like numpy recarrays.

    PupilData and many of its attributes are dataclasses, so all the Python dataclass
    functions can be applied to PupilData objects. The fields of the dataclass 
    correspond to the specific data items extracted by Pupil Player software during
    csv export, but there are additional attributes that can be accessed as well.

    Attributes
    ----------
    timestamp: float
        The Pupil software time at which the source image frame was captured in seconds.
    confidence: float
        The confidence in the pupil detection. 0 is the minimum confidence, 1 is the 
        maximum.
    norm_pos: data_structures.Cartesian2D
        The position of the center of the pupil in normalized coordinates in the eye
        camera image.
    diameter: float
        The diameter of the pupil in the camera image in pixels, not corrected for
        perspective.
    ellipse: data_structures.Ellipse
        The ellipse describing the shape of the pupil in the camera image. All members
        are given in camera image pixels for distances and degrees for angles.
    diameter_3d: float
        The estimated pupil diameter in millimetres. The Pupil Capture model assumes
        the eyeball has a specific diameter in order to estimate the pupil diameter in
        real units.
    sphere: data_structures.Sphere
        The sphere describing the 3D eye model. All units are in millimetres.
    circle_3d: data_structures.Circle3D
        The circle in 3D space mapped to the pupil of the 3D eye model. All units are
        in millimetres.
    theta: float
        The polar angle in radians from the center of the 3D eye model to the estimated
        pupil position.
    phi: float
        The azimuthal angle in radians from the center of the 3D eye model to the 
        estimated pupil position.
    projected_sphere: data_structures.ProjectedSphere
        The elliptical shape projected onto the camera image frame by the 3D eye model.
        All members are given in camera image pixels for distances and degrees for 
        angles.
    id: {0, 1}
        The eye ID, 0 for the participant's right eye, 1 for the left.
    topic: str
        The message topic associated with the data. The topic structure differs
        between main topics, but the general structure is 
        "[main topic].[subtopic].[subtopic]". For example: pupil.0.3d refers to a pupil
        data message for eye 0 using the 3d model method.
    method: str
        Describes which detector was used to detect the pupil.
    location: list[float]
        Latitude and longitude where data was captured, probably determined by IP
        address.
    model_confidence: float
        The confidence in the current eye model (between 0 and 1).
    world_index: int, optional
        The index of the closest world (front-facing) video frame.

    Examples
    --------
    The class initialization function is designed to take 3D pupil message data unpacked
    from a PLDATA file and use Python dictionary unpacking to fill in the parameters.

    >>> import msgpack
    >>> with open("./pupil.pldata", "rb") as f:
    >>>     unpacker = msgpack.Unpacker(f, use_list=False)
    >>>     for msg_topic, b_obj in unpacker:
    >>>         data = mpk.unpackb(b_obj)
    >>>         subtopics = msg_topic.split(".")
    >>>         main_topic = subtopics[0]
    >>>         method = subtopics[2]
    >>>         if main_topic == "pupil" and method == "3d":
    >>>             data = PupilData(**data)
    >>>             print(data)
    """
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
        """Create tuple only from fields, not all parameters/properties
        
        Create a nested tuple from all fields (note: not all members/object 
        properties!). Fields are a special feature of dataclasses, and in this case are 
        used to distinguish the "important" data from other information like metadata. 
        
        Any fields that are themselves dataclasses (e.g. Sphere, Ellipse) are also 
        converted to tuples, resulting in the nested tuple structure. The order of the
        fields in the tuple matches their definition order in the class. See 
        PupilData.to_numpy_recarray.

        Examples
        --------
        Example with the output expanded for better readability.

        >>> import msgpack
        >>> with open("./pupil.pldata", "rb") as f:
        >>>     unpacker = msgpack.Unpacker(f, use_list=False)
        >>>     for msg_topic, b_obj in unpacker:
        >>>         data = mpk.unpackb(b_obj)
        >>>         subtopics = msg_topic.split(".")
        >>>         main_topic = subtopics[0]
        >>>         method = subtopics[2]
        >>>         if main_topic == "pupil" and method == "3d":
        >>>             data = PupilData(**data)
        >>>             print(data.fields_to_tuple())
        >>>             break
            (
                345.96117200001027,
                -1,
                1.0,
                (0.24351970741892207, 0.44267443593140643),
                27.32890140708419,
                (
                    (46.75578382443304, 107.00650830116997), 
                    (21.31778407928547, 27.32890140708419), 
                    148.42380933109075
                ), 
                2.577837659311591,
                (
                    (0.26086557200170474, -2.328750597080022, 39.13150616664784), 
                    10.392304845413264
                ), 
                (
                    (-4.80148189549971, 1.1571421916810967, 30.751697721131602), 
                    (-0.48712461218222697, 0.3354301899929012, -0.806347443629383), 
                    1.2889188296557954
                ), 
                1.228734488061224,
                -2.1142342768113487,
                (
                    (97.74302257519636, 79.10477264712452), 
                    (180.19507618312036, 180.19507618312036), 
                    0.0
                )
            )
        """
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
        """Create tuple from fields, including the eye ID for CSV export

        CSV export lacks metadata and bundles data for both eyes in the same file, 
        therefore the eye ID needs to be included in the data columns.
        """
        field_values = list(self.fields_to_tuple())
        field_values.append(self.id)
        return tuple(field_values)


    def to_numpy_recarray(self) -> np.ndarray:
        """Create a numpy recarray from fields

        The datatype of the resulting array is the pupiltools.aliases.pupil_datatype.
        """
        data = self.fields_to_tuple()
        return np.array(data, dtype=pupil_datatype)


@dataclass(init=False)
class GazeData:
    """A class for handling gaze data from PLDATA files

    After unpacking, PLDATA files contain lists of messages where each message is
    a deeply nested set of dictionaries. This format is difficult to work with and
    prevents IDE tools like tab completion for easy object introspection. The GazeData
    class bundles this data into meaningful objects and allows easier conversion to
    useful datatypes like numpy recarrays.

    GazeData and many of its attributes are dataclasses, so all the Python dataclass
    functions can be applied to GazeData objects. The fields of the dataclass 
    correspond to the specific data items extracted by Pupil Player software during
    csv export, but there are additional attributes that can be accessed as well.

    Attributes
    ----------
    timestamp: float
        The Pupil software time at which the source image frame was captured in seconds.
    world_index: int, optional
        The index of the closest world (front-facing) video frame.
    confidence: float
        The confidence in the pupil detection. 0 is the minimum confidence, 1 is the 
        maximum.
    norm_pos: data_structures.Cartesian2D
        The gaze position in normalized coordinates in the front camera image.
    base_data: list[data_structures.PupilData]
        The eye data used to estimate the gaze position. Normally a list of length 2, 
        but when one pupil is not reliably detected only a single PupilData object is
        available.
    gaze_point_3d: data_structures.Cartesian3D
        The estimated gaze position in the 3D world (front) camera coordinate system.
    eye_centers_3d: list[data_structures.Cartesian3D]
        The estimated 3D position of each eye in the 3D world (front) camera coordinate
        system. In the event that one of the eyes was not detected properly, its
        coordinates are replaced with numpy NaNs.
    gaze_normals_3d: list[data_structures.Cartesian3D]
        The estimated 3D gaze vector from each eye in the 3D world (front) camera
        coordinate system. In the event that one of the eyes was not detected properly,
        its coordinates are replaced with numpy NaNs.
    Examples
    --------
    The class initialization function is designed to take 3D gaze message data unpacked
    from a PLDATA file and use Python dictionary unpacking to fill in the parameters.

    >>> import msgpack
    >>> with open("./gaze.pldata", "rb") as f:
    >>>     unpacker = msgpack.Unpacker(f, use_list=False)
    >>>     for msg_topic, b_obj in unpacker:
    >>>         data = mpk.unpackb(b_obj)
    >>>         subtopics = msg_topic.split(".")
    >>>         main_topic = subtopics[0]
    >>>         if main_topic == "gaze":
    >>>             data = GazeData(**data)
    >>>             print(data)
    """
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
        """Create tuple only from fields, not all parameters/properties
        
        Create a nested tuple from all fields (note: not all members/object 
        properties!). Fields are a special feature of dataclasses, and in this case are 
        used to distinguish the "important" data from other information like metadata. 
        
        Any fields that are themselves dataclasses (e.g. norm_pos) are also 
        converted to tuples, resulting in the nested tuple structure. The order of the
        fields in the tuple matches their definition order in the class.

        The base_data field is reduced to only the timestamp of the pupil data rather
        than the full data structure to match the behaviour of the CSV export from
        Pupil Labs' Pupil Player.

        Examples
        --------
        Example with the output expanded for better readability.

        >>> import msgpack
        >>> with open("./gaze.pldata", "rb") as f:
        >>>     unpacker = msgpack.Unpacker(f, use_list=False)
        >>>     for msg_topic, b_obj in unpacker:
        >>>         data = mpk.unpackb(b_obj)
        >>>         subtopics = msg_topic.split(".")
        >>>         main_topic = subtopics[0]
        >>>         if main_topic == "gaze":
        >>>             data = GazeData(**data)
        >>>             print(data.fields_to_tuple())
        >>>             break
            (
                345.9355710000091, 
                -1, 
                1.0, 
                (0.46460641707690487, 0.8831982210630259), 
                (345.9354560000065, 345.93568600001163),
                (-14.411410769452052, -117.89830813333006, 275.84140173241457),
                (
                    (19.98571631324934, 14.985869912781883, -19.92116119749855), 
                    (-40.022254973480145, 15.001520605051482, -19.949131602653416)
                ),
                (
                    (-0.10548705527122566, -0.39875290894360527, 0.9109712392711519),
                    (0.07872946880333444, -0.41731550273288076, 0.9053449298034135)
                )
            )
        """
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
        """Alias of GazeData.fields_to_tuple"""
        return self.fields_to_tuple()

    def to_numpy_recarray(self) -> np.ndarray:
        """Create a numpy recarray from fields

        The datatype of the resulting array is the pupiltools.aliases.gaze_datatype.
        """
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
"""List of strings naming each column for CSV export of pupil data"""

GAZE_CSV_FIELDS = _get_csv_header(GazeData)
"""List of strings naming each column for CSV export of gaze data"""


def flatten_nested_tuple(data: tuple[Any]) -> tuple[Any]:
    """Convert a nested tuple to a flat tuple with the same order of elements
    
    Recursively unpacks tuples of tuples to produce a single tuple with only basic
    elements (e.g ints, floats, strings).

    Parameters
    ----------
    data: tuple[Any]
        A tuple with tuple elements.
    
    Returns
    -------
    tuple[Any]
        A "flat" tuple with all nested tuples unpacked.
    
    Examples
    --------
    >>> data = (35, (3.6, "hello"), -5.3, (23, (5.6, 2.4)))
    >>> flat_data = flatten_nested_tuple(data)
    >>> print(flat_data)
        (35, 3.6, "hello", -5.3, 23, 5.6, 2.4)
    """
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
    print(test_p_data.fields_to_tuple())
    test_g_data = GazeData(**test_g_dict)
    print(test_g_data.fields_to_tuple())
