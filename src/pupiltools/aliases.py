from typing import TypeAlias
import numpy as np

AttributesType: TypeAlias = dict[str, str | np.float64 | np.int64]
TrialDataType: TypeAlias = dict[str, AttributesType | list[np.ndarray]]
RawParticipantDataType: TypeAlias = list[TrialDataType]
ResampledTrialDataType: TypeAlias = dict[str, AttributesType | np.ndarray]
ResampledParticipantDataType: TypeAlias = list[ResampledTrialDataType]

# NumPy datatype aliases
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