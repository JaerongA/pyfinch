"""
By Jaerong
Utility functions to make output directories & save output files
"""

from datetime import date
from database import load
from pathlib import Path


def make_dir(parent_path, *dir_name, add_date=True):
    """

    Parameters
    ----------
    parent_path : str
    dir_name : str
         (optional), if not exists, files will be saved in parent_dir
    add_date : bool
        make a sub-dir with a date

    Returns
    -------
    save_path : path
    """

    global save_path
    if dir_name:
        if add_date:
            today = date.today()
            save_path = parent_path / dir_name[0] / today.strftime("%Y-%m-%d")  # 2020-07-04
        else:
            save_path = parent_path / dir_name[0]
    else:
        if add_date:
            today = date.today()
            save_path = parent_path /  today.strftime("%Y-%m-%d")  # 2020-07-04
        else:
            save_path = parent_path

    # print(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    return save_path


def save_fig(fig, save_path, title, fig_ext='.png', open_folder=False):
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

    fig_name = save_path / (title + fig_ext)
    plt.savefig(fig_name, transparent=True)
    plt.close(fig)

    if open_folder:  # open folder after saving figures
        open_folder(save_path)


def save2json(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)
