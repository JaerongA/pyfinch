"""
By Jaerong
plot raster & peth per syllable
"""
def create_db():
    from database.load import ProjectLoader

    db = ProjectLoader().load_db()
    with open('database/create_syllable.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

def get_entropy(note_onsets, note_offsets, note_contexts, time_resolved):
    # Get mean entropy and variance
    from analysis.spike import AudioData

    entropy_mean = {}
    entropy_var = {}

    for context in ['U', 'D']:

        se_mean = np.array([], dtype=np.float32)
        se_var = np.array([], dtype=np.float32)
        ind = np.array(find_str(note_contexts, context))

        if ind.shape[0] >= nb_note_crit:
            for (start, end) in zip(note_onsets[ind], note_offsets[ind]):
                audio = AudioData(path).extract([start, end])  # audio object
                audio.spectrogram()  # get self.spect first
                se = audio.get_spectral_entropy(time_resolved=time_resolved)
                if type(se) == 'dict':
                    se_mean = np.append(se_mean, se['mean'])  # spectral entropy averaged over time bins per rendition
                    se_var = np.append(se_var, se['var'])  # spectral entropy variance per rendition
                else:
                    se_mean = np.append(se_mean, se)  # spectral entropy time-resolved
            entropy_mean[context] = round(se_mean.mean(), 3)
            if not time_resolved:
                entropy_var[context] = round(se_var.mean(), 3)
                return entropy_mean, entropy_var
            else:  # time-resolved version does not have entropy variance
                return entropy_mean

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from analysis.parameters import *
from analysis.spike import *
from database.load import DBInfo, ProjectLoader
from util import save
from util.draw import *
from analysis.parameters import pre_motor_win_size

# parameters
rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = 0.5  # text height
font_size = 12
marker_size = 0.4  # for spike count
nb_note_crit = 10  # minimum number of notes for analysis

# Gauss parameter for PETH smoothing
gauss_std = 3
filter_width = 20  # filter length for smoothing (in ms)
truncate = (((filter_width - 1)/2)-0.5)/ gauss_std

norm_method = None
fig_ext = '.png'  # .png or .pdf
update = False  # Set True for recreating a cache file
save_fig = True
update_db = True  # save results to DB
time_warp = True  # spike time warping
entropy = True  # calculate entropy & entropy variance
time_resolved = True  # computes time-resolved version of entropy

# Create & Load database
if update_db:
    db = create_db()

# Load database
# SQL statement
# Create a new database (syllable)
db = ProjectLoader().load_db()
# query = "SELECT * FROM cluster WHERE analysisOK=1"
query = "SELECT * FROM cluster WHERE id=96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    motif = cluster_db.motif

    # Load class object
    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object

    import copy

    syllables = ''.join(ci.syllables)
    onsets = np.hstack(ci.onsets)
    offsets = np.hstack(ci.offsets)
    durations = np.hstack(ci.durations)
    contexts = ''

    for i in range(len(ci.contexts)):  # concatenate contexts
        contexts += ci.contexts[i] * len(ci.syllables[i])

    # Loop through note
    for note in cluster_db.songNote:

        ind = np.array(find_str(syllables, note))  # note indices
        if not ind.size:  # if a note does not exist
            continue
        note_onsets = np.asarray(list(map(float, onsets[ind])))
        note_offsets = np.asarray(list(map(float, offsets[ind])))
        note_durations = np.asarray(list(map(float, durations[ind])))
        note_contexts = ''.join(np.asarray(list(contexts))[ind])
        note_median_dur = np.median(note_durations, axis=0)
        nb_note = len(ind)

        spk_ts = np.hstack(ci.spk_ts)

        note_spk_ts_list = []
        for onset, offset in zip(note_onsets, note_offsets):
            note_spk_ts_list.append(spk_ts[np.where((spk_ts >= onset - pre_motor_win_size) & (spk_ts <= offset))])

        # Perform piecewise linear warping
        note_spk_ts_warped_list = []

        for onset, duration, spk_ts in zip(note_onsets, note_durations, note_spk_ts_list):

            spk_ts_new = copy.deepcopy(spk_ts)
            ratio = note_median_dur / duration
            offset = 0
            origin = 0

            spk_ts_temp, ind = spk_ts[spk_ts >= onset], np.where(spk_ts >= onset)

            spk_ts_temp = ((ratio * ((spk_ts_temp - onset) )) + origin) + onset
            np.put(spk_ts_new, ind, spk_ts_temp)  # replace original spk timestamps with warped timestamps
            note_spk_ts_warped_list.append(spk_ts_new)

        # Nb_note per context
        nb_note = {}
        for context in ['U', 'D']:
            nb_note[context] = len(find_str(note_contexts, context))

        # Get note firing rates (includes pre-motor window) per context
        note_spk = {}
        note_fr = {}
        for context1 in ['U', 'D']:
            if nb_note[context1] >= nb_note_crit:
                note_spk[context1] = sum([len(spk) for context2, spk in zip(note_contexts, note_spk_ts_list) if context2==context1])
                note_fr[context1] = round(note_spk[context1] / ((note_durations[find_str(note_contexts, context1)] + pre_motor_win_size).sum() / 1E3), 3)
            else:
                note_fr[context1] = np.nan

        # Skip if there are not enough motifs per condition
        if np.prod([nb[1] < nb_note_crit for nb in nb_note.items()]):
            print("Not enough notes")
            continue

        # Plot spectrogram & peri-event histogram (Just the first rendition)

        # Note start and end
        start = note_onsets[0] - peth_parm['buffer']
        end = note_offsets[0] + peth_parm['buffer']
        duration = note_offsets[0] - note_onsets[0]

        # Get spectrogram
        audio = AudioData(path, update=update).extract([start, end])  # audio object
        audio.spectrogram(freq_range=freq_range)

        # Plot figure
        fig = plt.figure(figsize=(6,10))
        fig.set_tight_layout(False)
        note_name = ci.name + '-' + note
        if time_warp:
            fig_name = note_name + '  (time-warped)'
        else:
            fig_name = note_name + '  (non-warped)'
        plt.suptitle(fig_name, y=.93, fontsize=11)
        gs = gridspec.GridSpec(17, 5)
        gs.update(wspace=0.025, hspace=0.05)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3, 0:5])
        audio.spect_time = audio.spect_time - audio.spect_time[0] - peth_parm['buffer']  # starts from zero
        ax_spect.pcolormesh(audio.spect_time , audio.spect_freq, audio.spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_xlim(-peth_parm['buffer'], duration + peth_parm['buffer'])
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0, 0:5], sharex=ax_spect)
        onset = 0  # start from 0
        offset = onset + duration

        # Calculate spectral entropy per time bin
        # Plot syllable entropy
        if entropy:

            if time_resolved:
                entropy_mean = get_entropy(note_onsets, note_offsets, note_contexts)
            else:
                ax_se = ax_spect.twinx()
                se = audio.get_spectral_entropy(time_resolved=time_resolved)
                # time = audio.spect_time[np.where((audio.spect_time >= onset) & (audio.spect_time <= offset))]
                # se = se[np.where((audio.spect_time >= onset) & (audio.spect_time <= offset))]
                # ax_se.plot(time, se, 'k')
                ax_se.plot(audio.spect_time, se['array'], 'k')
                ax_se.set_ylim(0, 1)
                # se['array'] = se['array'][np.where((audio.spect_time >= onset) & (audio.spect_time <= offset))]
                remove_right_top(ax_se)
                # Calculate averaged entropy and entropy variance across renditions
                entropy_mean, entropy_var = get_entropy(note_onsets, note_offsets, note_contexts)

        # Mark syllables
        rectangle = plt.Rectangle((onset, rec_yloc), duration, 0.2,
                                  linewidth=1, alpha=0.5, edgecolor='k', facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
        ax_syl.add_patch(rectangle)
        ax_syl.text((onset + (offset - onset) / 2), text_yloc, note, size=font_size)
        ax_syl.axis('off')

        # Plot raster
        ax_raster = plt.subplot(gs[4:6, 0:5], sharex=ax_spect)
        line_offsets = np.arange(0.5, sum(nb_note.values()))
        if time_warp:
            zipped_lists = zip(note_contexts, note_spk_ts_warped_list, note_onsets)
        else:
            zipped_lists = zip(note_contexts, note_spk_ts_list, note_onsets)

        pre_context = ''  # for marking  context change
        context_change = np.array([])

        for note_ind, (context, spk_ts, onset) in enumerate(zipped_lists):

            spk = spk_ts - onset
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[note_ind],
                                linelengths=tick_length, linewidths=tick_width, orientation='horizontal')

            # Demarcate the note
            if time_warp:
                note_duration = note_median_dur
            else:
                note_duration = note_durations[note_ind]

            rectangle = plt.Rectangle((0, note_ind), note_duration, rec_height,
                                      fill=True,
                                      linewidth=1,
                                      alpha=0.15,
                                      facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
            ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=note_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, (note_ind))
                if pre_context:
                    ax_raster.text(ax_raster.get_xlim()[1] + 0.2,
                                   ((context_change[-1] - context_change[-2]) / 3) + context_change[-2],
                                   pre_context,
                                   size=6)
            pre_context = context

        # Demarcate the last block
        ax_raster.text(ax_raster.get_xlim()[1] + 0.2,
                       ((ax_raster.get_ylim()[1] - context_change[-1]) / 3) + context_change[-1],
                       pre_context,
                       size=6)

        ax_raster.set_yticks([0, sum(nb_note.values())])
        ax_raster.set_yticklabels([0, sum(nb_note.values())])
        ax_raster.set_ylim([0, sum(nb_note.values())])
        ax_raster.set_ylabel('Trial #', fontsize=font_size)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        remove_right_top(ax_raster)

        # Plot sorted raster
        ax_raster = plt.subplot(gs[7:9, 0:5], sharex=ax_spect)

        # Sort trials based on context
        sort_ind = np.array([i[0] for i in sorted(enumerate(note_contexts), key=lambda x: x[1], reverse=True)])
        contexts_sorted = np.array(list(note_contexts))[sort_ind].tolist()
        onsets_sorted = np.array(note_onsets)[sort_ind].tolist()
        if time_warp:
            spk_ts_sorted = np.array(note_spk_ts_warped_list)[sort_ind].tolist()
        else:
            spk_ts_sorted = np.array(note_spk_ts_list)[sort_ind].tolist()

        zipped_lists = zip(contexts_sorted, spk_ts_sorted, onsets_sorted)

        pre_context = ''  # for marking  context change
        context_change = np.array([])

        for note_ind, (context, spk_ts, onset) in enumerate(zipped_lists):

            # Plot rasters
            spk = spk_ts - onset
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[note_ind],
                                linelengths=tick_length, linewidths=tick_width, orientation='horizontal')

            # Demarcate the note
            if time_warp:
                note_duration = note_median_dur
            else:
                note_duration = note_durations[note_ind]

            rectangle = plt.Rectangle((0, note_ind), note_duration, rec_height,
                                      fill=True,
                                      linewidth=1,
                                      alpha=0.15,
                                      facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
            ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=note_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, (note_ind))
                if pre_context:
                    ax_raster.text(ax_raster.get_xlim()[1] + 0.2,
                                   ((context_change[-1] - context_change[-2]) / 3) + context_change[-2],
                                   pre_context,
                                   size=6)
            pre_context = context

        # Demarcate the last block
        ax_raster.text(ax_raster.get_xlim()[1] + 0.2,
                       ((ax_raster.get_ylim()[1] - context_change[-1]) / 3) + context_change[-1],
                       pre_context,
                       size=6)

        ax_raster.set_yticks([0, sum(nb_note.values())])
        ax_raster.set_yticklabels([0, sum(nb_note.values())])
        ax_raster.set_ylim([0, sum(nb_note.values())])
        ax_raster.set_ylabel('Trial #', fontsize=font_size)
        ax_raster.set_title('sorted raster', size=font_size)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        remove_right_top(ax_raster)

        # Draw peri-event histogram (PETH)
        # TODO: should get syllable object as a class

        from analysis.parameters import peth_parm
        import math
        import numpy as np

        peth = np.zeros((len(note_onsets), peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of trials x nb of time bins
        for trial_ind, (evt_ts, spk_ts) in enumerate(zip(note_onsets, note_spk_ts_warped_list)):
            evt_ts -= peth_parm['buffer']
            spk_ts -= evt_ts
            for spk in spk_ts:
                ind = math.ceil(spk / peth_parm['bin_size'])
                # print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
                peth[trial_ind, ind] += 1
        time_bin = peth_parm['time_bin'] - peth_parm['buffer']

        peth_dict = {}
        peth_dict['peth'] = peth
        peth_dict['time_bin'] = time_bin
        peth_dict['contexts'] = list(note_contexts)
        peth_dict['median_duration'] = note_median_dur

        # Get firing rates
        pi = PethInfo(peth_dict)
        # pi.get_fr(norm_method=norm_method)  # get firing rates
        from analysis.parameters import peth_parm, nb_note_crit
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        smoothing = True

        # Get trial-by-trial firing rates
        fr_dict = {}
        for k, v in pi.peth.items():  # loop through differ ent conditions in peth dict
            if v.shape[0] >= nb_note_crit:
                fr = v / (peth_parm['bin_size'] / 1E3)  # in Hz

                if smoothing:  # Gaussian smoothing
                    fr = gaussian_filter1d(fr, gauss_std, truncate=truncate)

                # Truncate values outside the range
                ind = (((0 - peth_parm['buffer']) <= time_bin) & (time_bin <= note_median_dur))
                fr = fr[:, ind]
                fr_dict[k] = fr
        fr = fr_dict
        time_bin = time_bin[ind]

        # Get mean firing rates
        mean_fr = {}
        for k, v in fr.items():
            fr = np.mean(v, axis=0)
            mean_fr[k] = fr

        # Plot mean firing rates
        ax_peth = plt.subplot(gs[10:12, 0:5], sharex=ax_spect)
        for context, fr in mean_fr.items():
            if context == 'U':
                ax_peth.plot(time_bin, fr, 'b', label=context)
            elif context == 'D':
                ax_peth.plot(time_bin, fr, 'm', label=context)

        plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend
        ax_peth.set_ylabel('FR', fontsize=font_size)

        fr_ymax = myround(round(ax_peth.get_ylim()[1], 3), base=5)
        ax_peth.set_ylim(0, fr_ymax)
        plt.yticks([0, ax_peth.get_ylim()[1]], [str(0), str(int(fr_ymax))])

        # Mark the baseline firing rates
        if 'baselineFR' in row.keys() and cluster_db.baselineFR:
            ax_peth.axhline(y=row['baselineFR'], color='k', ls='--', lw=0.5)

        # Mark end of the motif
        ax_peth.axvline(x=0, color='k', ls='--', lw=0.5)
        ax_peth.axvline(x=note_median_dur, color='k', lw=0.5)
        ax_peth.set_xlabel('Time (ms)')
        remove_right_top(ax_peth)

        # Calculate pairwise cross-correlation
        pcc_dict = {}
        for k, v in fr_dict.items():  # loop through different conditions in peth dict
            if k != 'All':
                if v.shape[0] >= nb_note_crit:
                    pcc = get_pcc(v)
                    pcc_dict[k] = pcc

        # Print out results on the figure
        txt_xloc = -0.5
        txt_yloc = 1
        txt_inc = 0.2  # y-distance between texts within the same section
        txt_offset = 0.25

        ax_txt = plt.subplot(gs[13:, 2])
        ax_txt.set_axis_off()  # remove all axes

        # # of motifs
        for i, (k, v) in enumerate(nb_note.items()):
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"# of notes ({k}) = {v}", fontsize=font_size)

        # Firing rates (includes the pre-motor window)
        for i, (k, v) in enumerate(note_fr.items()):
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"FR ({k}) = {v}", fontsize=font_size)

        # PCC
        txt_yloc -= txt_offset
        v = pcc_dict['U']['mean'] if "U" in pcc_dict else np.nan
        ax_txt.text(txt_xloc, txt_yloc, f"PCC (U) = {v}", fontsize=font_size)
        txt_yloc -= txt_inc

        v = pcc_dict['D']['mean'] if "D" in pcc_dict else np.nan
        ax_txt.text(txt_xloc, txt_yloc, f"PCC (D) = {v}", fontsize=font_size)

        # Corr context (correlation of firing rates between two contexts)
        txt_yloc -= txt_offset
        corr_context = None
        if 'U' in mean_fr.keys() and 'D' in mean_fr.keys():
            corr_context = round(np.corrcoef(mean_fr['U'], mean_fr['D'])[0, 1], 3)
        ax_txt.text(txt_xloc, txt_yloc, f"Context Corr = {corr_context}", fontsize=font_size)

        # Save results to database
        if update_db:   # only use values from time-warped data
            query = "INSERT OR IGNORE INTO " \
                    "syllable(clusterID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days, note)" \
                    "VALUES({}, '{}', '{}', {}, {}, {}, {}, {}, '{}')".format(cluster_db.id, cluster_db.birdID, cluster_db.taskName, cluster_db.taskSession,
                                                                              cluster_db.taskSessionDeafening, cluster_db.taskSessionPostDeafening,
                                                                              cluster_db.dph, cluster_db.block10days, note)
            db.cur.execute(query)

            if 'U' in nb_note:
                db.cur.execute(f"UPDATE syllable SET nbNoteUndir = ({nb_note['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'D' in nb_note:
                db.cur.execute(f"UPDATE syllable SET nbNoteDir = ({nb_note['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'U' in note_fr and nb_note['U'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable SET frUndir = ({note_fr['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
            if 'D' in note_fr and nb_note['D'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable SET frDir = ({note_fr['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'U' in pcc_dict and nb_note['U'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable SET pccUndir = ({pcc_dict['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'D' in pcc_dict and nb_note['D'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable SET pccDir = ({pcc_dict['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if corr_context:
                db.cur.execute(f"UPDATE syllable SET corrContext = ({corr_context}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if entropy:
                if 'U' in entropy_mean and nb_note['U'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable SET entropyUndir = ({entropy_mean['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in entropy_mean and nb_note['D'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable SET entropyDir = ({entropy_mean['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if time_resolved:
                    if 'U' in entropy_var and nb_note['U'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable SET entropyVarUndir = ({entropy_var['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                    if 'D' in entropy_var and nb_note['D'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable SET entropyVarDir = ({entropy_var['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
            db.conn.commit()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'RasterSyllable')
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
        else:
            plt.show()

# Convert db to csv
if update_db:
    db.to_csv('syllable')
print('Done!')
