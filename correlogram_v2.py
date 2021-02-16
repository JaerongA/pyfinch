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
import matplotlib.gridspec as gridspec
from pathlib import Path
from util import save


def print_out_text(ax, peak_latency,
                   firing_rates,
                   burst_mean_duration, burst_freq, burst_mean_spk, burst_fraction,
                   category,
                   burst_index
                   ):
    font_size = 12
    txt_xloc = 0
    txt_yloc = 1
    txt_inc = 0.15
    ax.set_ylim([0, 1])
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"Peak Latency = {round(peak_latency, 3)} ms", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"Firing Rates = {round(firing_rates, 3)} Hz", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Duration = {round(burst_mean_duration, 3)}', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Freq = {round(burst_freq, 3)}', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst MeanSpk = {round(burst_mean_spk, 3)}', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Fraction = {round(burst_fraction, 3)} (%)', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, 'Category = {}'.format(category), fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Index = {round(burst_index, 3)}', fontsize=font_size)

    ax.axis('off')


def update_db():
    pass


class BurstingInfo:

    def __init__(self, ClassInfo, *input_context):

        # ClassInfo can be BaselineInfo, MotifInfo etc
        if input_context:  # select data based on social context
            spk_list = [spk_ts for spk_ts, context in zip(ClassInfo.spk_ts, ClassInfo.contexts) if context == input_context[0]]
            duration_list = [duration for duration, context in zip(ClassInfo.durations, ClassInfo.contexts) if context == input_context[0]]
            self.context = input_context
        else:
            spk_list = ClassInfo.spk_ts
            duration_list = ClassInfo.durations

        # Bursting analysis
        burst_spk_list = []
        burst_duration_arr = []

        nb_bursts = []
        nb_burst_spk_list = []

        for ind, spks in enumerate(spk_list):

            # spk = bi.spk_ts[8]
            isi = np.diff(spks)  # inter-spike interval
            inst_fr = 1E3 / np.diff(spks)  # instantaneous firing rates (Hz)
            bursts = np.where(inst_fr >= burst_hz)[0]  # burst index

            # Skip if no bursting detected
            if not bursts.size:
                continue

            # Get the number of bursts
            temp = np.diff(bursts)[np.where(np.diff(bursts) == 1)].size  # check if the spikes occur in bursting
            nb_bursts = np.append(nb_bursts, bursts.size - temp)

            # Get burst onset
            temp = np.where(np.diff(bursts) == 1)[0]
            spk_ind = temp + 1
            # Remove consecutive spikes in a burst and just get burst onset

            burst_onset_ind = bursts

            for i, ind in enumerate(temp):
                burst_spk_ind = spk_ind[spk_ind.size - 1 - i]
                burst_onset_ind = np.delete(burst_onset_ind, burst_spk_ind)

            # Get burst offset index
            burst_offset_ind = np.array([], dtype=np.int)

            for i in range(bursts.size - 1):
                if bursts[i + 1] - bursts[i] > 1:  # if not successive spikes
                    burst_offset_ind = np.append(burst_offset_ind, bursts[i] + 1)

            # Need to add the subsequent spike time stamp since it is not included (burst is the difference between successive spike time stamps)
            burst_offset_ind = np.append(burst_offset_ind, bursts[bursts.size - 1] + 1)

            burst_onset = spks[burst_onset_ind]
            burst_offset = spks[burst_offset_ind]
            burst_spk_list.append(spks[burst_onset_ind[0]: burst_offset_ind[0] + 1])
            burst_duration_arr = np.append(burst_duration_arr, burst_offset - burst_onset)

            # Get the number of burst spikes
            nb_burst_spks = 1  # note that it should always be greater than 1

            if nb_bursts.size:
                if bursts.size == 1:
                    nb_burst_spks = 2
                    nb_burst_spk_list.append(nb_burst_spks)

                elif bursts.size > 1:
                    for ind in range(bursts.size - 1):
                        if bursts[ind + 1] - bursts[ind] == 1:
                            nb_burst_spks += 1
                        else:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
                            nb_burst_spks = 1

                        if ind == bursts.size - 2:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
            # print(nb_burst_spk_list)
        if sum(nb_burst_spk_list):
            self.spk_list = burst_spk_list
            self.nb_burst_spk = sum(nb_burst_spk_list)
            self.fraction = sum(nb_burst_spk_list) / sum([len(spks) for spks in spk_list])
            self.duration = (burst_duration_arr).sum()  # total duration
            self.freq = nb_bursts.sum() / sum(duration_list)
            self.mean_nb_spk = np.array(nb_burst_spk_list).mean()
            self.mean_duration = burst_duration_arr.mean()  # mean duration
        else:  # no burst spike detected
            self.spk_list = np.nan
            self.nb_burst_spk = np.nan
            self.fraction = np.nan
            self.duration = np.nan
            self.freq = np.nan
            self.mean_nb_spk = np.nan
            self.mean_duration = np.nan

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

# Parameter
normalize = False
update = False
nb_row = 7
nb_col = 3
fig_ext = '.png'  # .png or .pdf
save_fig = False
update_db = False  # save results to DB

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 5"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    mi = MotifInfo(row, update=update)  # motif object
    bi = BaselineInfo(row, update=update)  # baseline object

    # Calculate firing rates
    mi.get_mean_fr()

    # Get correlogram
    correlogram = mi.get_correlogram(mi.spk_ts, mi.spk_ts)
    correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)  # add baseline correlogram

    # Get correlogram object per condition
    if 'B' in correlogram.keys():
        corr_b = Correlogram(correlogram['B'])  # Baseline
    if 'U' in correlogram.keys():
        corr_u = Correlogram(correlogram['U'])  # Undirected
    if 'D' in correlogram.keys():
        corr_d = Correlogram(correlogram['D'])  # Directed

    # Get jittered correlogram for getting the baseline
    correlogram_jitter = mi.get_jittered_corr()
    correlogram_jitter['B'] = bi.get_jittered_corr()

    for key, value in correlogram_jitter.items():
        if key == 'U':
            corr_u.category(value)
        elif key == 'D':
            corr_d.category(value)
        elif key == 'B':
            corr_b.category(value)

    burst_info_b = BurstingInfo(bi)
    burst_info_u = BurstingInfo(mi, 'U')
    burst_info_d = BurstingInfo(mi, 'D')

    # # Bursting analysis
    # burst_spk_list = []
    # burst_duration_arr = []
    #
    # nb_bursts = []
    # nb_burst_spk_list = []
    #
    # for ind, spks in enumerate(bi.spk_ts):
    #
    #     # spk = bi.spk_ts[8]
    #     isi = np.diff(spks)  # inter-spike interval
    #     inst_fr = 1E3 / np.diff(spks)  # instantaneous firing rates (Hz)
    #     bursts = np.where(inst_fr >= burst_hz)[0]  # burst index
    #
    #     # Skip if no bursting detected
    #     if not bursts.size:
    #         continue
    #
    #     # Get the number of bursts
    #     temp = np.diff(bursts)[np.where(np.diff(bursts) == 1)].size  # check if the spikes occur in bursting
    #     nb_bursts = np.append(nb_bursts, bursts.size - temp)
    #
    #     # Get burst onset
    #     temp = np.where(np.diff(bursts) == 1)[0]
    #     spk_ind = temp + 1
    #     # Remove consecutive spikes in a burst and just get burst onset
    #
    #     burst_onset_ind = bursts
    #
    #     for i, ind in enumerate(temp):
    #         burst_spk_ind = spk_ind[spk_ind.size - 1 - i]
    #         burst_onset_ind = np.delete(burst_onset_ind, burst_spk_ind)
    #
    #     # Get burst offset index
    #     burst_offset_ind = np.array([], dtype=np.int)
    #
    #     for i in range(bursts.size - 1):
    #         if bursts[i + 1] - bursts[i] > 1:  # if not successive spikes
    #             burst_offset_ind = np.append(burst_offset_ind, bursts[i] + 1)
    #
    #     # Need to add the subsequent spike time stamp since it is not included (burst is the difference between successive spike time stamps)
    #     burst_offset_ind = np.append(burst_offset_ind, bursts[bursts.size - 1] + 1)
    #
    #     burst_onset = spks[burst_onset_ind]
    #     burst_offset = spks[burst_offset_ind]
    #
    #     # burst_spk_list.append(spks[burst_onset_ind[0]: burst_offset_ind[0] + 1])
    #     # burst_duration_list.append(burst_offset - burst_onset)
    #
    #     # burst_spk_arr = np.append(burst_spk, [spks[burst_onset_ind[0]:burst_offset_ind[0]]])
    #     burst_duration_arr = np.append(burst_duration_arr, [burst_offset - burst_onset])
    #
    #     # Get the number of burst spikes
    #     nb_burst_spks = 1  # note that it should always be greater than 1
    #
    #     if nb_bursts.size:
    #         if bursts.size == 1:
    #             nb_burst_spks = 2
    #             nb_burst_spk_list.append(nb_burst_spks)
    #
    #         elif bursts.size > 1:
    #             for ind in range(bursts.size - 1):
    #                 if bursts[ind + 1] - bursts[ind] == 1:
    #                     nb_burst_spks += 1
    #                 else:
    #                     nb_burst_spks += 1
    #                     nb_burst_spk_list.append(nb_burst_spks)
    #                     nb_burst_spks = 1
    #
    #                 if ind == bursts.size - 2:
    #                     nb_burst_spks += 1
    #                     nb_burst_spk_list.append(nb_burst_spks)
    #     # print(nb_burst_spk_list)
    #
    # # Calculate burst fraction
    # burst_fraction = sum(nb_burst_spk_list) / sum([len(spks) for spks in bi.spk_ts])
    # # print(burst_fraction)
    # burst_duration = (burst_duration_arr).sum()
    #
    # # Burst Frequency (number of burst / total sum of baseline period in Hz)
    # burst_freq = nb_bursts.sum() / sum(bi.durations)
    #
    # # Mean number of spikes per burst
    # burst_mean_spk = np.array(nb_burst_spk_list).mean()
    #
    # # Burst mean duration
    # burst_mean_duration = burst_duration_arr.mean()

    # Plot the results
    fig = plt.figure(figsize=(12, 7))
    fig.set_dpi(500)
    plt.suptitle(mi.name, y=.95)

    # The code will skip if a condition doesn't exist
    ax1 = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=3, colspan=1)
    if exists('corr_u'):
        ax2 = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=3, colspan=1)
    if exists('corr_d'):
        ax3 = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=3, colspan=1)

    # For text output
    ax_txt1 = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=3, colspan=1)
    if exists('corr_u'):
        ax_txt2 = plt.subplot2grid((nb_row, nb_col), (4, 1), rowspan=3, colspan=1)
    if exists('corr_u'):
        ax_txt3 = plt.subplot2grid((nb_row, nb_col), (4, 2), rowspan=3, colspan=1)

    # with suppress(NameError):
    #
    #     corr_b.plot_corr(ax1, spk_corr_parm['time_bin'], correlogram['B'], 'Baseline', normalize=normalize)
    #     print_out_text(ax_txt1, corr_b.peak_latency, bi.mean_fr, burst_mean_duration, burst_freq, burst_mean_spk,
    #                    burst_fraction, corr_b.category, corr_b.burst_index)
    #
    #     corr_u.plot_corr(ax2, spk_corr_parm['time_bin'], correlogram['U'], 'Undir', normalize=normalize)
    #     print_out_text(ax_txt2, corr_u.peak_latency, mi.mean_fr['U'], burst_mean_duration, burst_freq, burst_mean_spk,
    #                    burst_fraction, corr_u.category, corr_u.burst_index)
    #
    #     corr_d.plot_corr(ax3, spk_corr_parm['time_bin'], correlogram['D'], 'Dir', normalize=normalize)
    #     print_out_text(ax_txt3, corr_d.peak_latency, mi.mean_fr['D'], burst_mean_duration, burst_freq, burst_mean_spk,
    #                    burst_fraction, corr_d.category, corr_d.burst_index)

    plt.show()

    # save_path = save.make_dir('SpkCorr')
    # save.save_fig(fig, save_path, ci.name)

    # Save results to database
    # if update_db:
    #     with suppress(KeyError):
    #         db.create_col('cluster', 'baselineFR', 'REAL')
    #         db.update('cluster', 'baselineFR', row['id'], round(bi.mean_fr, 3))  # baseline firing rates
    #         db.create_col('cluster', 'motifFRUndir', 'REAL')
    #         db.update('cluster', 'motifFRUndir', row['id'],
    #                   round(mi.mean_fr['U'], 3))  # motif firing rates during Undir
    #         db.create_col('cluster', 'motifFRDir', 'REAL')
    #         db.update('cluster', 'motifFRDir', row['id'], round(mi.mean_fr['D'], 3))  # motif firing rates during Dir
    #
    # # Convert db to csv
    # if update_db:
    #     db.to_csv('cluster')
    # print('Done!')

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'UnitProfiling')
        save.save_fig(fig, save_path, mi.name, fig_ext=fig_ext)

# TODO: burst index, add database, save figure, isi plot
