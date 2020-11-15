"""
By Jaerong
A collection of utility functions used for analysis
"""

def unique(list):
    """
    Input : list
    Extract unique strings from the list in the order they appeared
    """
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


def find_str(pattern : str, string : str):
    """
    Find all indices of patterns in a string

    Parameters
    ----------
    pattern : str
        string pattern to search
    string : str
        input string

    Returns
    -------
    ind : list
        list of starting indices
    """
    import re
    if not pattern.isalpha(): # if the pattern contains non-alphabetic chars such as *
        pattern = "\\" + pattern

    ind = [m.start() for m in re.finditer(pattern, string)]
    return ind


def find_data_path():
    """Request the user to manually find dir path and return it"""
    from pathlib import Path
    from tkinter import Tk
    from tkinter import filedialog
    root = Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory()
    return Path(data_dir)