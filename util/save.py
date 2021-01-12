"""
By Jaerong
Utility functions to make output directories & save output files
"""

from pathlib import Path
from datetime import date
from database import load


def make_dir(parent_path, dir_name, add_date=True):
    """
    add date info in the sub-directory

    Args:
        parent_dir: path
        dir_name: str
        add_date: bool

    Returns:
        save_path: path
    """

    global save_path
    if add_date:
        today = date.today()
        save_path = parent_path / dir_name / today.strftime("%Y-%m-%d")  # 2020-07-04

    # print(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    return save_path


def save_fig(fig, save_path, title, ext='.png', open_folder=True):
    import matplotlib.pyplot as plt
    import matplotlib
    from util.functions import open_folder

    # Make the text in .pdf editable
    # pdf.fonttype : 42 # Output Type 3 (Type3) or Type 42 (TrueType)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Make Arial the default font
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    fig_name = save_path / (title + ext)
    plt.savefig(fig_name, transparent=True)
    plt.close(fig)

    if open_folder:  # open folder after saving figures
        open_folder(save_path)


def save2json(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)
