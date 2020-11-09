"""
By Jaerong
A collection of utility functions used for analysis
"""


def unique(list):
    """
    Inpur : list
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
