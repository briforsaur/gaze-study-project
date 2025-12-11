from argparse import ArgumentParser, Namespace
import json
import msgpack as mpk
from pathlib import Path
from statistics import mean, pstdev


_DEFAULT_OUTFILE_PATH = Path("./data.json")


def _get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("data_file_path", type=Path, help="Path to a PLDATA file.")
    parser.add_argument("out_file_path", type=Path, default=_DEFAULT_OUTFILE_PATH, help="Path to output JSON file.")
    return parser.parse_args()


def main(data_file_path: Path, out_file_path: Path = _DEFAULT_OUTFILE_PATH) -> None:
    with open(data_file_path, "rb") as data_file:
        unpacker = mpk.Unpacker(data_file, use_list=False)
        msg_topic: str
        msg_list = []
        data_list = []
        duration_list = []
        n_msg = 0
        for msg_topic, data_bytes in unpacker:
            n_msg += 1
            if msg_topic not in msg_list:
                msg_list.append(msg_topic)
            data = mpk.unpackb(data_bytes)
            data_list.append(data)
            if "surface" in msg_topic:
                t0 = data["gaze_on_surfaces"][0]["timestamp"]
                tf = data["gaze_on_surfaces"][-1]["timestamp"]
                duration_list.append(tf - t0)
    with open(out_file_path, "w") as f:
        json.dump(data_list, f, indent=2)
    assert float("NaN") not in duration_list
    print(f"Message duration mean: {mean(duration_list)}")
    print(f"Message duration std dev: {pstdev(duration_list)}")
    print(msg_list)
    print(f"Number of messages: {n_msg}")


if __name__ == "__main__":
    args = _get_args()
    main(**vars(args))
