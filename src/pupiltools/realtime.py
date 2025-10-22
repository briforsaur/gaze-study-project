from msgpack import unpackb, packb
import zmq
import numpy as np


_DEFAULT_FRAME_FORMAT = "bgr"
_DEFAULT_FRAME_SUBTOPICS = ["world", "eye.0", "eye.1"]


class PupilNetworkVideoHandler:

    def __init__(self, pupil_ip: str, pupil_port: str, subtopics: list[str] = _DEFAULT_FRAME_SUBTOPICS) -> None:
        self.subtopics = subtopics
        context = zmq.Context()
        # Open requests port
        self._req = context.socket(zmq.REQ)
        self._req.connect(f"tcp://{pupil_ip}:{pupil_port}")
        # Open subscription port
        self._req.send_string("SUB_PORT")
        sub_port = self._req.recv_string()
        self._sub = context.socket(zmq.SUB)
        self._sub.connect(f"tcp://{pupil_ip}:{sub_port}")
        # Set subscription topic
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "frame.")
        # Tell the Pupil Network API the desired frame format
        notification = {
            "subject": "frame_publishing.set_format",
            "format": _DEFAULT_FRAME_FORMAT
        }
        self.notify(notification)

    def notify(self, notification: dict[str, str]) -> str:
        topic = "notify." + notification["subject"]
        payload = packb(notification, use_bin_type=True)
        self._req.send_string(topic, flags=zmq.SNDMORE)
        self._req.send(payload)
        return self._req.recv_string()
    
    def recv_from_sub(self) -> tuple[str, dict]:
        topic = self._sub.recv_string()
        payload = unpackb(self._sub.recv(), raw=False)
        extra_frames = []
        while self._sub.get(zmq.RCVMORE):
            extra_frames.append(self._sub.recv())
        if extra_frames:
            payload["__raw_data__"] = extra_frames
        return topic, payload
    
    def has_new_data_available(self):
        return self._sub.get(zmq.EVENTS) & zmq.POLLIN # type: ignore
    
    def get_latest_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        while self.has_new_data_available():
            # Continue collecting frames from the buffer until none are left
            topic, payload = self.recv_from_sub()
            main_topic, subtopic = topic.split(".", maxsplit=1)
            if main_topic == "frame" and payload["format"] != _DEFAULT_FRAME_FORMAT:
                print(f"different frame format ({payload['format']}), skipping frame from {topic}.")
            elif main_topic == "frame" and subtopic in self.subtopics:
                latest_frame = np.frombuffer(payload["__raw_data__"][0], dtype=np.uint8)
                # Frame arrives as 1-D array, needs to be reshaped to H, W, and BGR channels
                latest_frame = latest_frame.reshape(payload["height"], payload["width"], 3)
                frames[subtopic] = latest_frame
        return frames