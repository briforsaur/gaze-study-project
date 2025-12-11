from argparse import ArgumentParser, Namespace
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path


_DEFAULT_OUTFILE_PATH = Path("./temp/surface_gaze_tracking.pdf")
SURFACE_DIMENSIONS = (150, 200)  # millimetres


def _get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "data_file_path", type=Path, help="Path to a JSON surface data file."
    )
    parser.add_argument(
        "out_file_path",
        type=Path,
        help="Path to save the output file.",
        default=_DEFAULT_OUTFILE_PATH,
    )
    return parser.parse_args()


def main(data_file_path: Path, out_file_path: Path = _DEFAULT_OUTFILE_PATH) -> None:
    with open(data_file_path, mode="r") as data_file:
        data_list = json.load(data_file)
    t0 = data_list[0]["gaze_on_surfaces"][0]["timestamp"]
    gaze_positions = []
    for surface_datum in data_list:
        for gaze_datum in surface_datum["gaze_on_surfaces"]:
            # Scaling the normalized position to SURFACE_DIMENSIONS unit (millimetres)
            gaze_position = [
                norm_pos * scale
                for norm_pos, scale in zip(gaze_datum["norm_pos"], SURFACE_DIMENSIONS)
            ]
            gaze_positions.append([gaze_datum["timestamp"] - t0, *gaze_position])
    gaze_positions = np.array(gaze_positions)
    t = gaze_positions[:, 0]
    fig, ax = plt.subplots()
    ax.plot(gaze_positions[:1200, 1], gaze_positions[:1200, 2])
    # Labelling gaze points at integer second time points
    for t_label in range(1, int(t[:1200].max())):
        label_idx = np.argmax(t > t_label)
        ax.annotate(
            f"{t_label}", tuple(gaze_positions[label_idx, 1:]), backgroundcolor="w"
        )
    # Adding a rectangle to show the surface region
    ax.add_patch(
        Rectangle(
            (0, 0), *SURFACE_DIMENSIONS, edgecolor="k", linestyle="-", facecolor="none"
        )
    )
    ax.axis("equal")
    ax.set_xlim(-75, 225)
    ax.set_ylim(-100, 300)
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title("Gaze Position Relative to Surface")
    fig.savefig(out_file_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = _get_args()
    main(**vars(args))
