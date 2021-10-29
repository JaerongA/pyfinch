"""
Utility functions to make output directories & save output files
"""


def make_dir(parent_path, *dir_name, add_date=True):
    """
    Make a new directory
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
    from datetime import date

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
            save_path = parent_path / today.strftime("%Y-%m-%d")  # 2020-07-04
        else:
            save_path = parent_path

    # print(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    return save_path


def save_fig(fig, save_path, title, fig_ext='.png', view_folder=False, dpi=None):
    """
    Function for saving figures
    Parameters
    ----------
    fig : figure object
    save_path : path
        directory path to save figures
    title : str
        title of the figure
    fig_ext : str
        figure extension (e.g., '.pdf' for vector output), '.png' by default
    view_folder : bool
        open the folder where the figure is saved
    dpi : int
        increase the value for enhanced resolution
    """

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
    plt.savefig(fig_name, transparent=True, dpi=dpi)
    plt.close(fig)

    if view_folder:  # open folder after saving figures
        open_folder(save_path)


def save2json(filename, data):
    """
    Save data in .json format
    Parameters
    ----------
    filename : str
    data : arr
    """
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)
