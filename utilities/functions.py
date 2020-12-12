"""
By Jaerong
A collection of utility functions used for analysis
"""

def unique(list):
    """
    Extract unique strings from the list in the order they appear

    Args:
        list: list
            list of strings

    Returns:
        list:
            list of unique, ordered strings
    """
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


def find_str(pattern : str, string : str):
    """
    Find all indices of patterns in a string

    Args:
        pattern: str
            string pattern to search
        string: str
            input string

    Returns:
        ind : list
            list of starting index values
    """
    import re
    if not pattern.isalpha(): # if the pattern contains non-alphabetic chars such as *
        pattern = "\\" + pattern

    ind = [m.start() for m in re.finditer(pattern, string)]
    return ind


def find_data_path():
    """
    Request the user to manually find dir path and return it
    """
    from pathlib import Path
    from tkinter import Tk
    from tkinter import filedialog
    root = Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory()
    return Path(data_dir)


def list_files(dir: str, ext: str):
    """
    Return the list of files in the current directory
        Input1: directory name (path)
        Input2: file extension (str) (e.g., .wav, .rhd etc)
        Output: list of file path (list)
    """
    files = [file for file in dir.rglob('*' + ext)]
    return files


def open_folder(dir: str):
    """
    Open the directory in win explorer
        Input: directory name (path)
    """
    import webbrowser
    webbrowser.open(dir)