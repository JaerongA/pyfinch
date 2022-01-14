"""
Helper functions for drawing a save_fig
"""


def set_fig_size(w, h, ax=None):
    """ w, h: width, height in inches """
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
    """Remove top and right axis"""
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)


def get_ax_lim(ax, base=10):
    """
    Get axis limit
    Parameters
    ----------
    ax : axis object
    base : int
        default = 10
    """
    from math import ceil, floor

    ax_min, ax_max = ax.get_ylim()[0], ax.get_ylim()[1]
    ax_min = floor(ax_min * base) / base
    ax_max = ceil(ax_max * base) / base
    return ax_min, ax_max
