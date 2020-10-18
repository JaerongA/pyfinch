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
