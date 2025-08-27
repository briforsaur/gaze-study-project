PupilTools API
==============

.. currentmodule:: pupiltools

Constants
---------
.. currentmodule:: pupiltools.constants
.. autosummary::
    
    PARTICIPANTS
    TASK_TYPES
    TS_FILE_SUFFIX
    DATA_FILE_SUFFIX

Exporting data
--------------
.. currentmodule:: pupiltools.export
.. autosummary::
    :toctree: generated/

    export_data_csv
    export_folder
    export_hdf
    export_hdf_from_raw
    extract_data
    get_metadata
    get_subfolders_from_log

Data Structures
---------------
.. currentmodule:: pupiltools.data_structures
.. autosummary::
    :toctree: generated/

    PupilData
    GazeData
    Cartesian2D
    Cartesian3D
    Axes
    Ellipse
    Sphere
    Circle3D
    ProjectedSphere
    flatten_nested_tuple
    PUPIL_CSV_FIELDS
    GAZE_CSV_FIELDS

Type Aliases
------------
.. currentmodule:: pupiltools.aliases
.. autosummary::
    :toctree: generated/

    AttributesType
    TrialDataType
    RawParticipantDataType
    ResampledTrialDataType
    ResampledParticipantDataType
    planar_position_dt
    position_dt
    axes_dt
    ellipse_dt
    sphere_dt
    circle_3d_dt
    projected_sphere_dt
    eyes_base_dt
    eye_coords_dt
    pupil_datatype
    gaze_datatype