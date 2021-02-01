"""
By Jaerong
Plot spike correlograms
"""

from analysis.spike import *
from analysis.parameters import *
from analysis.load import read_rhd
from contextlib import suppress
from database.load import ProjectLoader
import matplotlib.pyplot as plt
from pathlib import Path
from util import save


def plot_correlogram(ax, time_bin, correlogram, title, font_size=10, normalize=False):
    """
    Plot correlogram
    Parameters
    ----------
    ax : axis to plot the figure
    time_bin : array
    correlogram : array
    title : str
    font_size : title font size
    normalize : normalize the correlogram
    """
    from util.draw import remove_right_top

    ax.bar(time_bin, correlogram, color='k')

    ymax = myround(ax.get_ylim()[1], base=5)
    ax.set_ylim(0, ymax)
    plt.yticks([0, ax.get_ylim()[1]], [str(0), str(int(ymax))])
    ax.set_title(title, size=font_size)
    ax.set_xlabel('Time (ms)')
    if normalize:
        ax.set_ylabel('Prob')
    else:
        ax.set_ylabel('Count')
    remove_right_top(ax)

# Parameter
font_size=10
normalize=False

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    ci = ClusterInfo(row)  # cluster object
    bi = BaselineInfo(row)  # baseline object

    correlogram = ci.get_correlogram(ci.spk_ts, ci.spk_ts)
    correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)

    # Analysis on the correlogram
    # Todo : peak latency, burst fraction, category, burst inex, burst mean spk, burst duration, burst freq











    # Plot the results
    fig = plt.figure(figsize=(12, 4))
    plt.text(0.5, 1.08, ci.name,
             horizontalalignment='center',
             fontsize=20)

    with suppress(KeyError):
        ax = plt.subplot(131)
        plot_correlogram(ax, spk_corr_parm['time_bin'], correlogram['B'], 'Baseline', normalize=normalize)

        ax = plt.subplot(132)
        plot_correlogram(ax, spk_corr_parm['time_bin'], correlogram['U'], 'Undir', normalize=normalize)
        ax.set_ylabel('')

        ax = plt.subplot(133)
        plot_correlogram(ax, spk_corr_parm['time_bin'], correlogram['D'], 'Dir', normalize=normalize)
        ax.set_ylabel('')





    plt.show()

    # save_path = save.make_dir('SpkCorr')
    # save.save_fig(fig, save_path, ci.name)


