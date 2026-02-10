# Copyright 2026 Shane Forbrigger
# Licensed under the MIT License (see LICENSE file in project root)

import cv2 as cv
from itertools import pairwise
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict


_CV_FONT = cv.FONT_HERSHEY_SIMPLEX


class FramePayload(TypedDict):
    topic: str
    width: int
    height: int
    index: int
    timestamp: float
    format: str
    __raw_data__: list[bytes]


def image_array_from(frame_payload: FramePayload):
    img_array = np.frombuffer(frame_payload["__raw_data__"][0], dtype=np.uint8)
    # Frame arrives as 1-D array, needs to be reshaped to H, W, and BGR channels
    return img_array.reshape(frame_payload["height"], frame_payload["width"], 3)


def display_camera_frames(
    frames: dict[str, NDArray[np.uint8]],
    norm_gaze_pos: tuple[float, float],
    surface_to_img_homography: list[list[float]] | None = None,
):
    """Display camera frames in separate, labelled windows with gaze annotations"""
    for label, image_array in frames.items():
        # Copy is required because original array is not editable
        image_array = np.copy(image_array)
        if label == "world":
            image_dims = image_array.shape[:2]
            xy_pos = gaze_position_to_cv_frame(norm_gaze_pos, image_dims)  # type: ignore
            annotate_world_frame(image_array, xy_pos)
            if surface_to_img_homography is not None:
                draw_surface_bounding_box(image_array, surface_to_img_homography)
        elif label == "eye.0":
            rotate_image(image_array)
        cv.imshow(label, image_array)


def annotate_world_frame(image_array: NDArray[np.uint8], gaze_coords: NDArray[np.int_]):
    """Add gaze point circle and gaze point pixel coordinates to a camera image"""
    xy_pos_str = f"[{gaze_coords[0]:4d}, {gaze_coords[1]:4d}]"
    cv.circle(image_array, tuple(gaze_coords), 25, (0, 0, 255), 2)
    cv.putText(
        image_array, xy_pos_str, (0, 50), _CV_FONT, 1, (0, 0, 255), 3, cv.LINE_AA
    )


def draw_surface_bounding_box(
    image_array: NDArray[np.uint8], surface_to_img_homography: list[list[float]]
):
    homography_matrix = np.array(surface_to_img_homography)
    corner_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    # Get a range of points between each corner to show the true bounding box
    # Due to camera distortion, the border isn't necessarily a straight line b/t corners
    surface_point_ranges = [
        np.linspace([start], [end], num=11, axis=1)[
            :, :-1, :
        ]  # Skip the last point because it is included in the next range
        for start, end in pairwise(corner_points)
    ]
    surface_points = np.concatenate(surface_point_ranges, axis=1)
    image_points = cv.perspectiveTransform(surface_points, homography_matrix)
    n_points = surface_points.shape[1]
    for i in range(n_points):
        # Get the current and next x,y point, dropping the 3rd element artefact from the
        # homographic transformation
        points = (
            tuple(image_points[0, i, :2].astype(int)),
            tuple(image_points[0, (i + 1) % n_points, :2].astype(int)),
        )
        line_colour = (255, 0, 0)  # Blue
        cv.line(image_array, points[0], points[1], line_colour, 1)


def gaze_position_to_cv_frame(
    norm_pos: tuple[float, float], img_dims: tuple[int, int]
) -> NDArray[np.int_]:
    """Convert normalized image frame coordinates to image pixel coordinates"""
    img_array = np.array([img_dims[1], img_dims[0]])
    frame_pos = np.array([0, 1]) + np.array([1, -1]) * np.array(norm_pos)
    frame_pos = img_array * frame_pos
    return frame_pos.astype(int)


def rotate_image(image_array: NDArray[np.uint8], angle: float = 180.0):
    """Rotate an image about its center while keeping the same height and width"""
    image_dims = image_array.shape[:2]
    image_center = (np.array([image_dims[1], image_dims[0]]) - 1) / 2.0
    rotation_matrix = cv.getRotationMatrix2D(tuple(image_center), angle, scale=1)
    cv.warpAffine(image_array, rotation_matrix, image_dims, dst=image_array)