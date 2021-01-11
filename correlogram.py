from database.load import ProjectLoader

from analysis.spike import *
from analysis.parameters import *
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
from util import save

query = "SELECT * FROM cluster WHERE id == 96"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)

for row in cur.fetchall():

    ci = ClusterInfo(row)
    #
    # ci._load_events()
    # ci._load_spk()

    bi = BaselineInfo(row)

    correlogram = ci.get_correlogram(ci.spk_ts, ci.spk_ts)

    correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)


    # Plot the results
    #TODO plot the results
    fig = plt.figure(figsize=(12, 4))
    # plt.title(ci.name, size=10, y=1.08)
    plt.text(0.5, 1.08, ci.name,
             horizontalalignment='center',
             fontsize=20)
    ax = plt.subplot(131)
    ax.bar(spk_corr_parm['time_bin'], correlogram['B'], color='k')
    ax.set_title('Baseline', size=10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Prob')
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    ax = plt.subplot(132)
    ax.bar(spk_corr_parm['time_bin'], correlogram['U'], color='k')
    ax.set_title('Undir', size=10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Prob')
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    ax = plt.subplot(133)
    ax.bar(spk_corr_parm['time_bin'], correlogram['D'], color='k')
    ax.set_title('Dir', size=10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Prob')
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # save_path = save.make_dir('SpkCorr')
    # save.save_fig(fig, save_path, ci.name)
