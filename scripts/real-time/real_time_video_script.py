from argparse import ArgumentParser, Namespace
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from dataclasses import astuple

from pupiltools.data_structures import GazeData
from pupiltools.realtime import PupilNetworkHandler
from pupiltools.video import image_array_from, display_camera_frames

_DEFAULT_PUPIL_IP = "127.0.0.1"
_DEFAULT_PUPIL_PORT = "50020"


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


if __name__ == "__main__":
    args = _get_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
