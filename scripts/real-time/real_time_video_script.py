# Script based on https://github.com/pupil-labs/pupil-helpers/blob/master/python/recv_world_video_frames_with_visualization.py
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np

from pupiltools.realtime import PupilNetworkVideoHandler


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


def main(pupil_ip: str=_DEFAULT_PUPIL_IP, pupil_port: str=_DEFAULT_PUPIL_PORT):
    """Quit by typing 'q'"""
    pupil_net_handler = PupilNetworkVideoHandler(pupil_ip, pupil_port)
    frames: dict[str, np.ndarray] = {}
    print(
        "To stop the script, press 'q' while one of the video windows is selected,",
        "or type CTRL+C in the terminal window.")
    while True:
        frames.update(pupil_net_handler.get_latest_frames())
        if all(subtopic in frames.keys() for subtopic in pupil_net_handler.subtopics):
            # All 3 cameras have delivered an image
            for label, image_array in frames.items():
                cv2.imshow(label, image_array)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    args = _get_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
