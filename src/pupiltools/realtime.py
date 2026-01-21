from msgpack import unpackb, packb
from typing import Any
import zmq


_DEFAULT_FRAME_FORMAT = "bgr"
_DEFAULT_FRAME_SUBTOPICS = ["world", "eye.0", "eye.1"]
_DEFAULT_TOPICS = ["gaze", "frame"]


class PupilNetworkHandler:

    def __init__(self, pupil_ip: str, pupil_port: str, topics: list[str] = _DEFAULT_TOPICS) -> None:
        self.subtopics = [topic.removeprefix("frame.") for topic in topics if "frame" in topic]
        # TODO: Fix default case of subtopics when only "frame" is in topics
        if not self.subtopics and "frame" in topics:
            self.subtopics = _DEFAULT_FRAME_SUBTOPICS
        context = zmq.Context()
        # Open requests port
        self._req = context.socket(zmq.REQ)
        self._req.connect(f"tcp://{pupil_ip}:{pupil_port}")
        # Open subscription port
        self._req.send_string("SUB_PORT")
        sub_port = self._req.recv_string()
        self._sub = context.socket(zmq.SUB)
        self._sub.connect(f"tcp://{pupil_ip}:{sub_port}")
        self._subscribe_to(topics)
        # Tell the Pupil Network API the desired frame format
        notification = {
            "subject": "frame_publishing.set_format",
            "format": _DEFAULT_FRAME_FORMAT
        }
        self._notify(notification)

    def _subscribe_to(self, topics: list[str]):
        if "pupil" in topics:
            # Subscribe only to 3d pupil data
            topics.remove("pupil")
            topics.extend([f"pupil.{i}.3d" for i in range(2)])
        for topic in topics:
            self._sub.setsockopt_string(zmq.SUBSCRIBE, f"{topic}")

    def _notify(self, notification: dict[str, str]) -> str:
        topic = "notify." + notification["subject"]
        payload = packb(notification, use_bin_type=True)
        self._req.send_string(topic, flags=zmq.SNDMORE)
        self._req.send(payload)
        return self._req.recv_string()
    
    def _recv_from_sub(self) -> tuple[str, dict[str, Any]]:
        topic = self._sub.recv_string()
        payload = unpackb(self._sub.recv(), raw=False)
        extra_frames = []
        while self._sub.get(zmq.RCVMORE):
            extra_frames.append(self._sub.recv())
        if extra_frames:
            payload["__raw_data__"] = extra_frames
        return topic, payload
    
    def _has_new_data_available(self):
        return self._sub.get(zmq.EVENTS) & zmq.POLLIN # type: ignore
    
    def get_latest_data(self) -> dict[str, dict[str, Any]]:
        latest_data: dict[str, dict[str, Any]] = {}
        while self._has_new_data_available():
            # Continue collecting data from the buffer until none are left
            topic, payload = self._recv_from_sub() 
            latest_data.update({topic: payload})
        return latest_data