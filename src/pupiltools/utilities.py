import numpy as np
from typing import Generic, TypeVar
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mpl_fig
from datetime import date
from pathlib import Path


def fix_datetime_string(dt_str: str) -> str:
    """Fixes datetime strings with a typo to match ISO format
    
    The datetime strings recorded in the experiment had a typo in the separator between
    the date and the time such that they do not match ISO format.
    Correct ISO format: YYYY-MM-DDTHH:MM:SS
    Typo:               YYYY-MM-DD-THH:MM:SS

    This function fixes the typo and returns a properly formatted ISO datetime string.
    """
    dt_str_parts = dt_str.split("-")
    return "-".join(dt_str_parts[0:-1]) + dt_str_parts[-1]


def make_digit_str(num: int, width: int = 3) -> str:
    """Convert an integer to a fixed-width string padded with zeros

    Examples: 
    >>> make_digit_str(1)
    '001'
    >>> make_digit_str(1, 4)
    '0001'
    >>> make_digit_str(12, 4)
    '0012'
    """
    return "{:0={width}}".format(num, width=width)


def save_figure(fig: mpl_fig, fig_path: Path, figname: str, formats: str | list[str], verbose = False):
    """Save a matplotlib figure

    Saves a matplotlib figure in a single or multiple formats. Automatically prepends
    the date to the filename, i.e. the output file is f"YYYY-MM-DD_{figname}.{format}".
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure handle to be saved.
    fig_path: pathlib.Path
        Folder/directory to save the figure.
    figname: str
        Desired name for the figure after the date suffix.
    formats: str | list[str]
        Desired file format to use. Can be a single string like "png" or an iterable of
        strings like ("png", "svg"). The strings must be a format supported by 
        matplotlib's savefig() method.
    """
    datestr = date.today().isoformat()
    if isinstance(formats, str):
        formats = (formats,)
    for format in formats:
        full_path = fig_path / f"{datestr}_{figname}.{format}"
        fig.savefig(full_path)
        if verbose:
            print(f"Saved figure: {full_path}")


# Helper class used as a type for matplotlib arrays of Axes
# Based on https://stackoverflow.com/a/74197401
T = TypeVar('T')
class ObjArray(np.ndarray, Generic[T]):
    def __getitem__(self, key) -> T:
        return super().__getitem__(key)

if __name__=="__main__":
    typo_str = "2024-09-01-T12:00:00"
    print(f"Typo string: {typo_str}")
    print(f"Corrected:   {fix_datetime_string(typo_str)}")
    print(make_digit_str(1))
    print(make_digit_str(10))
    print(make_digit_str(100))
    print(make_digit_str(100, 5))