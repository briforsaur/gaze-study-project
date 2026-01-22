from argparse import ArgumentParser, Namespace
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict
from dataclasses import astuple

from pupiltools.data_structures import GazeData
from pupiltools.realtime import PupilNetworkHandler


_DEFAULT_PUPIL_IP = "127.0.0.1"
_DEFAULT_PUPIL_PORT = "50020"
_CV_FONT = cv.FONT_HERSHEY_SIMPLEX


def _get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pupil_ip",
        default=_DEFAULT_PUPIL_IP,
        help="IPv4 address that the Pupil Network plugin is broadcasting on.",
    )
    parser.add_argument(
        "--pupil_port",
        default=_DEFAULT_PUPIL_PORT,
        help="Network port that the Pupil Network plugin is broadcasting on.",
    )
    return parser.parse_args()


def main(pupil_ip: str = _DEFAULT_PUPIL_IP, pupil_port: str = _DEFAULT_PUPIL_PORT):
    """Quit by typing 'q'"""
    topics = ["gaze", "frame.world", "surfaces"]
    pupil_net_handler = PupilNetworkHandler(pupil_ip, pupil_port, topics)
    frames: dict[str, NDArray[np.uint8]] = {}
    print(
        "To stop the script, press 'q' while one of the video windows is selected,",
        "or type CTRL+C in the terminal window.",
    )
    norm_gaze_pos = (0.0, 0.0)
    surface_to_img_homography = None
    while True:
        latest_data = pupil_net_handler.get_latest_data()
        # Create dict of frames where the keys are the subtopics
        latest_frames = {
            topic.removeprefix("frame."): image_array_from(payload)  # type: ignore
            for topic, payload in latest_data.items()
            if "frame" in topic
        }
        gaze_data = None
        t = 0
        # Get most recent gaze datum (latest data can have gaze.3d.{0, 1, 01})
        for topic, data in latest_data.items():
            if "gaze" in topic:
                if data["timestamp"] > t:
                    gaze_data = GazeData(**data)
                    t = data["timestamp"]
        latest_surfaces = {
            topic.removeprefix("surfaces."): payload  # type: ignore
            for topic, payload in latest_data.items()
            if "surfaces" in topic
        }
        if latest_surfaces is not None:
            surface = latest_surfaces.get("Surface 1")
            if surface is not None:
                surface_to_img_homography = surface.get("surf_to_dist_img_trans")
        frames.update(latest_frames)
        if gaze_data is not None:
            norm_gaze_pos = astuple(gaze_data.norm_pos)
        display_camera_frames(frames, norm_gaze_pos, surface_to_img_homography)
        if cv.waitKey(1) == ord("q"):
            break


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
    surface_to_img_homography: list[list[float]] | None,
):
    """Display camera frames in separate, labelled windows with gaze annotations"""
    for label, image_array in frames.items():
        # Copy is required because original array is not editable
        image_array = np.copy(image_array)
        if label == "world":
            image_dims = image_array.shape[:2]
            xy_pos = gaze_position_to_cv_frame(norm_gaze_pos, image_dims)  # type:ignore
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
    surface_points = np.array(
        [[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32
    )
    image_points = cv.perspectiveTransform(surface_points, homography_matrix)
    for i in range(surface_points.shape[1]):
        point = tuple(image_points[0, i, :2].astype(int))
        colour = (255, int(255 * (3 - i) / 3), 0)
        text_position = (0, 100 + 50 * i)
        cv.circle(image_array, point, 10, colour, 2)
        xy_pos_str = f"[{point[0]:4d}, {point[1]:4d}]"
        cv.putText(
            image_array,
            xy_pos_str,
            text_position,
            _CV_FONT,
            1,
            colour,
            3,
            cv.LINE_AA,
        )


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


if __name__ == "__main__":
    args = _get_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
