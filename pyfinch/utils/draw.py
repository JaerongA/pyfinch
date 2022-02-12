"""
Helper functions for drawing a figure
"""


def set_fig_size(w, h, ax=None):
    """
    set size of a figure

    Parameters
    ----------
    w : float
        width in inches
    h : float
        height in inches
    ax : axis object
    """
    import matplotlib.pyplot as plt

    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def remove_right_top(ax):
    """
    Remove top and right axis

    Parameters
    ----------
    ax : axis object
    """
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)


def get_ax_lim(ax_min, ax_max, base=10):
    """
    Get axis limit

    Parameters
    ----------
    ax_min : float
    ax_max : float
    base : int
        default = 10
    """
    from math import ceil, floor

    ax_min = floor(ax_min * base) / base
    ax_max = ceil(ax_max * base) / base
    return ax_min, ax_max
