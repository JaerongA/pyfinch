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
update = False

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # ci = ClusterInfo(row, update=update)  # cluster object
    # correlogram = ci.get_correlogram(ci.spk_ts, ci.spk_ts)

    mi = MotifInfo(row, update=update)  # motif object
    correlogram = mi.get_correlogram(mi.spk_ts, mi.spk_ts)

    bi = BaselineInfo(row, update=update)  # baseline object
    correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)

    # Analysis on the correlogram
    # Todo : peak latency, burst fraction, category, burst inex, burst mean spk, burst duration, burst freq

    # correlogram = correlogram['U']
    # corr_center = round(correlogram.shape[0] / 2) + 1
    # corr_peak = np.argmax(correlogram)
    # peak_latency = np.min(np.abs(np.argwhere(correlogram == np.amax(correlogram)) - corr_center))  # in ms
    # (corr_center - (1000 / burst_crit)), (corr_center + (1000 / burst_crit))

    # correlogram = correlogram['B']
    corr_b = Correlogram(correlogram['B'])  # Load correlogram object


    # Plot the results
    fig = plt.figure(figsize=(12, 4))
    fig.set_dpi(500)
    plt.text(0.5, 1.08, mi.name,
             horizontalalignment='center',
             fontsize=20)

    with suppress(KeyError):
        ax = plt.subplot(131)
        corr_b.plot_corr(ax, corr_b.time_bin, corr_b.data, 'Baseline', normalize=normalize)

    #     ax = plt.subplot(132)
    #     plot_correlogram(ax, spk_corr_parm['time_bin'], correlogram['U'], 'Undir', normalize=normalize)
    #     ax.set_ylabel('')
    #
    #     ax = plt.subplot(133)
    #     plot_correlogram(ax, spk_corr_parm['time_bin'], correlogram['D'], 'Dir', normalize=normalize)
    #     ax.set_ylabel('')



    #
    #
    # plt.show()

    # save_path = save.make_dir('SpkCorr')
    # save.save_fig(fig, save_path, ci.name)



    # Bursting analysis
    burst_spk_list = []
    burst_duration_list = []

    nb_bursts = np.array([], dtype=np.int)
    nb_burst_spk_list = []

    for ind, spks in enumerate(bi.spk_ts):

        # spk = bi.spk_ts[8]
        isi = np.diff(spks)  # inter-spike interval
        inst_fr = 1E3 /np.diff(spks)  #  instantaneous firing rates (Hz)
        bursts = np.where(inst_fr >= burst_hz)[0]  # burst index

        # Skip if no bursting detected
        if not bursts.size:
            continue

        # Get the number of bursts
        temp = np.diff(bursts)[np.where(np.diff(bursts) == 1)].size  #  check if the spikes occur in bursting
        nb_bursts = np.append(nb_bursts, bursts.size - temp)

        # Get burst onset
        temp = np.where(np.diff(bursts) == 1)[0]
        spk_ind = temp + 1
        # Remove consecutive spikes in a burst and just get burst onset

        burst_onset_ind = bursts

        for i, ind in enumerate(temp):
            burst_spk_ind = spk_ind[spk_ind.size -1 - i]
            burst_onset_ind = np.delete(burst_onset_ind, burst_spk_ind)


        # Get burst offset index
        burst_offset_ind = np.array([], dtype=np.int)

        for i in range(bursts.size-1):
            if bursts[i+1] - bursts[i] > 1:  # if not successive spikes
                burst_offset_ind = np.append(burst_offset_ind, bursts[i] + 1)

        # Need to add the subsequent spike time stamp since it is not included (burst is the difference between successive spike time stamps)
        burst_offset_ind = np.append(burst_offset_ind, bursts[bursts.size - 1] + 1)

        burst_onset = spks[burst_onset_ind]
        burst_offset = spks[burst_offset_ind]

        burst_spk_list.append(spks[burst_onset_ind[0] : burst_offset_ind[0] + 1])
        burst_duration_list.append(burst_offset - burst_onset)

        # burst_spk = np.append(burst_spk, [spks[burst_onset_ind[0]:burst_offset_ind[0]]])
        # burst_duration = np.append(burst_duration, [burst_offset - burst_onset])


        # Get the number of burst spikes
        nb_burst_spks = 1  # note that it should always be greater than 1

        if nb_bursts.size:
            if bursts.size == 1:
                nb_burst_spks = 2
                nb_burst_spk_list.append(nb_burst_spks)

            elif  bursts.size > 1:
                for ind in range(bursts.size-1):
                    if bursts[ind+1] - bursts[ind] == 1:
                        nb_burst_spks += 1
                    else:
                        nb_burst_spks += 1
                        nb_burst_spk_list.append(nb_burst_spks)
                        nb_burst_spks = 1

                    if ind == bursts.size - 2:
                        nb_burst_spks += 1
                        nb_burst_spk_list.append(nb_burst_spks)
        print(nb_burst_spk_list)