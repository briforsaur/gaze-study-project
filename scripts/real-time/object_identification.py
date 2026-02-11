from argparse import ArgumentParser, Namespace
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from torch import accelerator
from ultralytics.models import YOLO

from pupiltools.realtime import PupilNetworkHandler
from pupiltools.video import image_array_from

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
    topics = ["frame.world",]
    device = get_torch_device()
    print("Loading YOLO model...")
    model = YOLO("yolo11n.pt")
    print("Finished.")
    # Sometimes the user forgets to start Pupil Capture before starting the script
    print("Waiting for Pupil connection. Has Pupil Capture started?", end="")
    pupil_net_handler = PupilNetworkHandler(pupil_ip, pupil_port, topics)
    # Clear previous console output and print that the connection was successful
    print("\x1b[1K\rPupil connection established.")
    print(
        "To stop the script, press 'q' while one of the video windows is selected,",
        "or type CTRL+C in the terminal window.",
    )
    frames: dict[str, NDArray[np.uint8]] = {}
    while True:
        latest_data = pupil_net_handler.get_latest_data()
        # Create dict of frames where the keys are the subtopics
        latest_frames = {
            topic.removeprefix("frame."): image_array_from(payload)  # type: ignore
            for topic, payload in latest_data.items()
            if "frame" in topic
        }
        frames.update(latest_frames)
        if 'world' in frames:
            results = model.predict(frames['world'], device=device, verbose=False)
            annotated_frame = results[0].plot()
            cv.imshow("world", annotated_frame)
        if cv.waitKey(1) == ord("q"):
            break
        

def get_torch_device() -> str:
    if accelerator.is_available():
        device = accelerator.current_accelerator().type # type: ignore
    else:
        device = "cpu"
    return device


if __name__ == "__main__":
    args = _get_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
