from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pupiltools.data_import as di
from pupiltools.utilities import make_digit_str
from pupiltools.constants import PARTICIPANTS, TASK_TYPES


def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the folder containing HDF Files.")
    parser.add_argument("--fig_path", type=Path, default=None, help="Path to the folder to save figures.")
    return parser.parse_args()


def main(data_path: Path, fig_path: Path = None): #type: ignore
    file_paths = [data_path / f"{p}.hdf5" for p in PARTICIPANTS]
    # Load all instruction times from HDF files
    #   Keep organized by task type
    instruction_times = get_instruction_times(file_paths)
    for task, t_list in instruction_times.items():
        assert isinstance(task, str)
        # Convert lists to ndarrays for use in numpy stats functions
        instruction_times[task] = np.array(t_list) # type: ignore
    # Calculate mean and standard deviation
    mean = {key: np.mean(t_ins) for key, t_ins in instruction_times.items()}
    std_dev = {key: np.std(t_ins) for key, t_ins in instruction_times.items()}
    for key in mean.keys():
        print(key.capitalize())
        print(f"Mean   : {mean[key]} s")
        print(f"Std Dev: {std_dev[key]} s")
    # Plot histogram
    ax = plt.subplot()
    ax.hist(instruction_times["action"])
    ax.hist(instruction_times["observation"])
    ax.legend(TASK_TYPES)
    ax.set_xlabel("Time of Instruction End [s]")
    ax.set_ylabel("Number of Trials")
    if fig_path is not None:
        if not fig_path.exists(): fig_path.mkdir(parents=True)
        plt.gcf().savefig(fig_path / "instruction-times-histogram.pdf", bbox_inches="tight")
    plt.show()
    


def get_instruction_times(file_paths: list[Path]) -> dict[str, list[float]]:
    instruction_times = {task: [] for task in TASK_TYPES}
    for file_path in file_paths:
        data, _ = di.get_resampled_participant_data(file=file_path, variables="timestamp")
        for trial_entry in data:
            t_ins = trial_entry["attributes"]["t_instruction"]
            assert isinstance(t_ins, float)
            task = trial_entry["attributes"]["task"]
            assert isinstance(task, str)
            # Some instruction times were not properly updated, 
            if t_ins < 5.0:
                instruction_times[task].append(t_ins)
    return instruction_times


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))