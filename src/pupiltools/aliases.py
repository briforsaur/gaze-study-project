# Copyright 2025 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

from typing import TypeAlias
import numpy as np

AttributesType: TypeAlias = dict[str, str | np.float64 | np.int64]
"""Type alias for Attributes entries in HDF files

Each trial has a set of metadata contained in the HDF attributes. This type alias
represents a dictionary holding that metadata. For example::
    {
        "die": 4,
        "recording": '003',
        "t_instruction": 348.8,
        "t_start": 345.9,
        "t_stop": 349.5,
        "task": 'observation',
        "trial": 0,
    }
"""

TrialDataType: TypeAlias = dict[str, AttributesType | list[np.ndarray]]
"""Type alias for the trial data returned from raw data HDF files

Each trial in a dataset has metadata and data. The metadata is associated with the
``"attributes"`` key and is of type ``AttributesType``. The data is associated with the 
``"data"`` key and is a 2-element list (one for each eye) of numpy record arrays of type
``pupil_datatype``.
"""

RawParticipantDataType: TypeAlias = list[TrialDataType]
"""Type alias for a list of ``TrialDataType``"""

ResampledTrialDataType: TypeAlias = dict[str, AttributesType | np.ndarray]
"""Type alias for the trial data returned from resampled data HDF files

Similar to ``TrialDataType``, except that after resampling the numpy record arrays for 
each eye in the ``"data"`` field are combined into a single, multidimensional numpy 
array rather than being multiple entries in a list.
"""

ResampledParticipantDataType: TypeAlias = list[ResampledTrialDataType]
"""Type alias for a list of ``ResampledTrialDataType``"""

# NumPy datatype aliases
planar_position_dt = np.dtype([("x", np.double), ("y", np.double)])
"""A numpy record array dtype for x and y position"""

position_dt = np.dtype([("x", np.double), ("y", np.double), ("z", np.double)])
"""A numpy record array dtype for x, y, z position"""

axes_dt = np.dtype([("a", np.double), ("b", np.double)])
"""A numpy array dtype for elliptical axes, semi-major ``a`` and semi-minor ``b``"""

ellipse_dt = np.dtype(
    [("center", planar_position_dt), ("axes", axes_dt), ("angle", np.double)]
)
"""A numpy record array dtype for an ellipse in 2D

A general ellipse described by the 2D position of its ``"center"``, the lengths of its
major and minor ``"axes"``, and the ``"angle"`` from the horizontal to the major axis.
"""

sphere_dt = np.dtype([("center", position_dt), ("radius", np.double)])
"""A numpy record array dtype for a sphere"""

circle_3d_dt = np.dtype(
    [("center", position_dt), ("normal", position_dt), ("radius", np.double)]
)
"""A numpy record array dtype for a circle in 3D space"""

projected_sphere_dt = np.dtype(
    [("center", planar_position_dt), ("axes", axes_dt), ("angle", np.double)]
)
"""A numpy record array dtype for the ellipse created by the projection of a sphere"""

eyes_base_dt = np.dtype([("eye0", np.double), ("eye1", np.double)])
"""A numpy record array dtype for the timestamps for each eye used in gaze estimation"""

eye_coords_dt = np.dtype([("eye0", position_dt), ("eye1", position_dt)])
"""A numpy record array dtype for the 3D position for each eye in gaze estimation"""


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
"""A numpy record array dtype to represent pupil data

The fields/keys of this dtype match the fields/column names in a Pupil Labs' Pupil
Player export of Pupil Core data.
"""


gaze_datatype = np.dtype(
    [
        ("timestamp", np.double),
        ("world_index", np.longlong),
        ("confidence", np.double),
        ("norm_pos", planar_position_dt),
        ("base_data", eyes_base_dt),
        ("gaze_point_3d", position_dt),
        ("eye_centers_3d", eye_coords_dt),
        ("gaze_normals_3d", eye_coords_dt)
    ]
)
"""A numpy record array dtype to represent gaze data

The fields/keys of this dtype match the fields/column names in a Pupil Labs' Pupil
Player export of Pupil Core data.
"""