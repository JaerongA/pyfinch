"""
plot raster & peth per syllable
"""

from analysis.functions import get_spectral_entropy
from analysis.parameters import freq_range, peth_parm, note_color, tick_width, tick_length, nb_note_crit
from analysis.spike import ClusterInfo, AudioData, pcc_shuffle_test
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
from database.load import create_db, DBInfo, ProjectLoader
from util import save
from util.draw import remove_right_top
from util.functions import find_str, myround
import warnings
warnings.filterwarnings('ignore')


def get_raster_syllable():

    # Create & load database
    if update_db:
        db = create_db('create_syllable_pcc.sql')

    # parameters
    rec_yloc = 0.05
    rec_height = 1  # syllable duration rect
    text_yloc = 0.5  # text height
    font_size = 10
    marker_size = 0.4  # for spike count

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
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
        audio = AudioData(path)  # audio object

        # Loop through note
        if TARGET_NOTE == 'all':
            notes = cluster_db.songNote
        else:
            notes = TARGET_NOTE

        for note in notes:
            # Load note object
            ni = ci.get_note_info(note, pre_buffer=pre_buffer, post_buffer=post_buffer)  # this will be used for plotting raster
            if not ni:  # the target note does not exist
                print("The note does not exist!")
                continue

            # Select context
            if NOTE_CONTEXT:
                ni.select_context(target_context=NOTE_CONTEXT)

            # Skip if there are not enough motifs per condition
            if np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()]):
                print("Not enough notes!")
                continue


            # Plot figure
            fig = plt.figure(figsize=(7, 10), dpi=500)
            fig.set_tight_layout(False)
            note_name = ci.name + '-' + note
            if time_warp:
                fig_name = note_name + '  (time-warped)'
            else:
                fig_name = note_name + '  (non-warped)'
            plt.suptitle(fig_name, y=.93, fontsize=11)
            gs = gridspec.GridSpec(17, 5)
            gs.update(wspace=0.025, hspace=0.05)

            # Plot spectrogram & peri-event histogram (Just the first rendition)
            # Note start and end
            start = ni.onsets[0] - peth_parm['buffer']
            end = ni.offsets[0] + peth_parm['buffer']
            duration = ni.durations[0]

            # Get spectrogram
            # Load audio object with info from .not.mat files
            timestamp, data = audio.extract([start, end])
            spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)
            del timestamp, data

            # Plot spectrogram
            ax_spect = plt.subplot(gs[1:3, 0:5])
            spect_time = spect_time - spect_time[0] - peth_parm['buffer']  # starts from zero
            ax_spect.pcolormesh(spect_time, spect_freq, spect,  # data
                                cmap='hot_r', rasterized=True,
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

            # Calculate spectral entropy per time bin
            # Plot syllable entropy
            if entropy:
                # Calculate spectral entropy
                entropy_mean = ni.get_entropy(mode='spectral')
                # Calculate averaged entropy and entropy variance across renditions
                spectro_temporal_entropy_mean, entropy_var = ni.get_entropy(mode='spectro_temporal')

                # Plot entropy over time (includes buffer window)
                ax_se = ax_spect.twinx()
                se = get_spectral_entropy(spect, mode='spectro_temporal')
                ax_se.plot(spect_time, se['array'], 'k', linewidth=3)
                ax_se.set_ylim(0, 1)
                ax_se.spines['top'].set_visible(False)
            del spect

            # Plot syllable duration
            ax_syl = plt.subplot(gs[0, 0:5], sharex=ax_spect)
            onset = 0  # start from 0
            offset = onset + duration

            # Mark syllables
            rectangle = plt.Rectangle((onset, rec_yloc), duration, 0.2,
                                      linewidth=1, alpha=0.5, edgecolor='k',
                                      facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onset + (offset - onset) / 2), text_yloc, note, size=font_size)
            ax_syl.axis('off')

            # Plot raster
            ax_raster = plt.subplot(gs[4:6, 0:5], sharex=ax_spect)
            line_offsets = np.arange(0.5, sum(ni.nb_note.values()))

            if time_warp:
                zipped_lists = zip(ni.contexts, ni.spk_ts_warp, ni.onsets)
            else:
                zipped_lists = zip(ni.contexts, ni.spk_ts, ni.onsets)

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
                    note_duration = ni.median_dur
                else:
                    note_duration = ni.durations[note_ind]

                rectangle = plt.Rectangle((0, note_ind), note_duration, rec_height,
                                          fill=True,
                                          linewidth=1,
                                          alpha=0.15, rasterized=True,
                                          facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
                ax_raster.add_patch(rectangle)

                # Demarcate song block (undir vs dir) with a horizontal line
                if pre_context != context:
                    ax_raster.axhline(y=note_ind, color='k', ls='-', lw=0.3)
                    context_change = np.append(context_change, note_ind)
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

            ax_raster.set_yticks([0, sum(ni.nb_note.values())])
            ax_raster.set_yticklabels([0, sum(ni.nb_note.values())])
            ax_raster.set_ylim([0, sum(ni.nb_note.values())])
            ax_raster.set_ylabel('Trial #', fontsize=font_size)
            plt.setp(ax_raster.get_xticklabels(), visible=False)
            remove_right_top(ax_raster)

            # Plot sorted raster
            if not NOTE_CONTEXT:
                ax_raster = plt.subplot(gs[7:9, 0:5], sharex=ax_spect)

                # Sort trials based on context
                sort_ind = np.array([i[0] for i in sorted(enumerate(ni.contexts), key=lambda x: x[1], reverse=True)])
                contexts_sorted = np.array(list(ni.contexts))[sort_ind].tolist()
                # ni.onsets = note_onsets
                onsets_sorted = np.array(ni.onsets)[sort_ind].tolist()
                if time_warp:
                    spk_ts_sorted = np.array(ni.spk_ts_warp)[sort_ind].tolist()
                else:
                    # ni.spk_ts = note_spk_ts_list
                    spk_ts_sorted = np.array(ni.spk_ts)[sort_ind].tolist()

                zipped_lists = zip(contexts_sorted, spk_ts_sorted, onsets_sorted)

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
                        note_duration = ni.median_dur
                    else:
                        note_duration = ni.durations[note_ind]

                    rectangle = plt.Rectangle((0, note_ind), note_duration, rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15,
                                              facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
                    ax_raster.add_patch(rectangle)

                    # Demarcate song block (undir vs dir) with a horizontal line
                    if pre_context != context:
                        ax_raster.axhline(y=note_ind, color='k', ls='-', lw=0.3)
                        context_change = np.append(context_change, note_ind)
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

                ax_raster.set_yticks([0, sum(ni.nb_note.values())])
                ax_raster.set_yticklabels([0, sum(ni.nb_note.values())])
                ax_raster.set_ylim([0, sum(ni.nb_note.values())])
                ax_raster.set_ylabel('Trial #', fontsize=font_size)
                ax_raster.set_title('Sorted raster', size=font_size)
                plt.setp(ax_raster.get_xticklabels(), visible=False)
                remove_right_top(ax_raster)

            # Draw peri-event histogram (PETH)
            pi = ni.get_note_peth(duration=note_duration)  # PETH object (PethInfo)
            pi.get_fr()  # get firing rates

            # Plot mean firing rates
            ax_peth = plt.subplot(gs[10:12, 0:5], sharex=ax_spect)
            for context, fr in pi.mean_fr.items():
                if context == 'U':
                    ax_peth.plot(pi.time_bin, fr, 'b', label=context)
                elif context == 'D':
                    ax_peth.plot(pi.time_bin, fr, 'm', label=context)

            plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend
            ax_peth.set_ylabel('FR', fontsize=font_size)

            fr_ymax = myround(round(ax_peth.get_ylim()[1], 3), base=5)
            ax_peth.set_ylim(0, fr_ymax)
            plt.yticks([0, ax_peth.get_ylim()[1]], [str(0), str(int(fr_ymax))])

            # Mark the baseline firing rates
            if 'baselineFR' in row.keys() and cluster_db.baselineFR:
                ax_peth.axhline(y=row['baselineFR'], color='k', ls='--', lw=0.5)

            # Mark syllable duration
            ax_peth.axvline(x=0, color='k', ls='--', lw=0.5)
            ax_peth.axvline(x=ni.median_dur, color='k', lw=0.5)
            ax_peth.set_xlabel('Time (ms)')
            remove_right_top(ax_peth)
            # Calculate metrics
            # pairwise cross-correlation
            pi.get_pcc()

            # Get shuffled PETH & pcc
            if shuffled_baseline:
                p_sig = pcc_shuffle_test(ni, pi, plot_hist=plot_hist)

            # Calculate sparseness index
            sparseness = pi.get_sparseness(bin_size=3)

            # CV of firing rates (over time)
            fr_cv = pi.get_fr_cv()

            # Get spike count fano factor
            pi.get_spk_count()  # pi.spk_count, pi.fano_factor, pi.spk_count_cv added

            # Print out results on the figure
            txt_xloc = -2.5
            txt_yloc = 0.8
            txt_inc = 0.15  # y-distance between texts within the same section

            ax_txt = plt.subplot(gs[13:, 2])
            ax_txt.set_axis_off()  # remove all axes

            # # of notes
            for i, (k, v) in enumerate(ni.nb_note.items()):
                ax_txt.text(txt_xloc, txt_yloc, f"# of notes ({k}) = {v}", fontsize=font_size)
                txt_yloc -= txt_inc
            txt_yloc -= txt_inc

            # Firing rates (includes the pre-motor window)
            # Load NoteInfo class again to calculate firing rates from a different window
            ni = ci.get_note_info(note, pre_buffer=pre_buffer, post_buffer=post_buffer)  # this will be used for plotting raster

            if NOTE_CONTEXT:
                ni.select_context(target_context=NOTE_CONTEXT)

            for i, (k, v) in enumerate(ni.mean_fr.items()):
                if v is not np.nan:
                    ax_txt.text(txt_xloc, txt_yloc, f"FR ({k}) = {v}", fontsize=font_size)
                    txt_yloc -= txt_inc
            txt_yloc -= txt_inc

            # PCC
            if "U" in pi.pcc and ni.nb_note['U'] >= nb_note_crit:
                t = ax_txt.text(txt_xloc, txt_yloc, f"PCC (U) = {pi.pcc['U']['mean']}", fontsize=font_size)
                if "p_sig" in locals():
                    if 'U' in p_sig and p_sig['U']:
                        t.set_bbox(dict(facecolor='green', alpha=0.5))
                    else:
                        t.set_bbox(dict(facecolor='red', alpha=0.5))
            txt_yloc -= txt_inc

            if "D" in pi.pcc and ni.nb_note['D'] >= nb_note_crit:
                t = ax_txt.text(txt_xloc, txt_yloc, f"PCC (D) = {pi.pcc['D']['mean']}", fontsize=font_size)
                if "p_sig" in locals():
                    if 'D' in p_sig and p_sig['D']:
                        t.set_bbox(dict(facecolor='green', alpha=0.5))
                    else:
                        t.set_bbox(dict(facecolor='red', alpha=0.5))
            txt_yloc -= txt_inc

            # Corr context (correlation of firing rates between two contexts)
            corr_context = None
            if 'U' in pi.mean_fr.keys() and 'D' in pi.mean_fr.keys():
                corr_context = round(np.corrcoef(pi.mean_fr['U'], pi.mean_fr['D'])[0, 1], 3)
            ax_txt.text(txt_xloc, txt_yloc, f"Context Corr = {corr_context}", fontsize=font_size)
            txt_yloc -= txt_inc

            txt_xloc = -0.5
            txt_yloc = 0.8

            # Sparseness index
            for context, value in sorted(sparseness.items(), reverse=True):
                if ni.nb_note[context] >= nb_note_crit:
                    ax_txt.text(txt_xloc, txt_yloc, f"Sparseness ({context}) = {value}", fontsize=font_size)
                txt_yloc -= txt_inc
            txt_yloc -= txt_inc

            # CV of firing rates (over time)
            for context, cv in sorted(fr_cv.items(), reverse=True):
                if ni.nb_note[context] >= nb_note_crit:
                    ax_txt.text(txt_xloc, txt_yloc, f"CV of FR ({context}) = {cv}", fontsize=font_size)
                txt_yloc -= txt_inc
            txt_yloc -= txt_inc

            # Fano factor (spike counts)
            for context, fano_factor in sorted(pi.fano_factor.items(), reverse=True):
                if ni.nb_note[context] >= nb_note_crit:
                    fano_factor = round(np.nanmean(pi.fano_factor[context]), 3)
                    ax_txt.text(txt_xloc, txt_yloc, f"Fano Factor ({context}) = {fano_factor}", fontsize=font_size)
                txt_yloc -= txt_inc
            txt_yloc -= txt_inc

            # Syllable duration
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"Duration = {note_duration : 0.3f} (ms)", fontsize=font_size)

            # Print out syllable entropy (if exists)
            if entropy:
                txt_xloc = 1.8
                txt_yloc = 0.8
                for context, value in entropy_mean.items():
                    ax_txt.text(txt_xloc, txt_yloc, f"Entropy ({context}) = {value}", fontsize=font_size)
                    txt_yloc -= txt_inc
                txt_yloc -= txt_inc

                for context, value in spectro_temporal_entropy_mean.items():
                    ax_txt.text(txt_xloc, txt_yloc, f"ST Entropy ({context}) = {value}", fontsize=font_size)
                    txt_yloc -= txt_inc
                txt_yloc -= txt_inc

                for context, value in entropy_var.items():
                    ax_txt.text(txt_xloc, txt_yloc, f"EV ({context}) = {value}", fontsize=font_size)
                    txt_yloc -= txt_inc
                txt_yloc -= txt_inc

            # Save results to database
            if update_db:  # only use values from time-warped data
                sql = "INSERT OR IGNORE INTO " \
                        "syllable_pcc (clusterID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days, note)" \
                        "VALUES({}, '{}', '{}', {}, {}, {}, {}, {}, '{}')".format(cluster_db.id, cluster_db.birdID,
                                                                                  cluster_db.taskName,
                                                                                  cluster_db.taskSession,
                                                                                  cluster_db.taskSessionDeafening,
                                                                                  cluster_db.taskSessionPostDeafening,
                                                                                  cluster_db.dph,
                                                                                  cluster_db.block10days,
                                                                                  note)
                db.cur.execute(sql)

                if 'U' in ni.nb_note:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET nbNoteUndir = ({ni.nb_note['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in ni.nb_note:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET nbNoteDir = ({ni.nb_note['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'U' in ni.mean_fr and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET frUndir = ({ni.mean_fr['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                if 'D' in ni.mean_fr and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET frDir = ({ni.mean_fr['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'U' in pi.pcc and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET pccUndir = ({pi.pcc['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in pi.pcc and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET pccDir = ({pi.pcc['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if corr_context:
                    db.cur.execute(
                        f"UPDATE syllable_pcc SET corrContext = ({corr_context}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                # Add sparseness index
                db.create_col('syllable_pcc', 'sparsenessUndir', 'REAL')
                db.create_col('syllable_pcc', 'sparsenessDir', 'REAL')
                db.conn.commit()

                for context in sparseness.keys():
                    if ni.nb_note[context] >= nb_note_crit:
                        col_name = 'sparsenessUndir' if context == 'U' else 'sparsenessDir'
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET {col_name} = ({sparseness[context]}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                # Add CV of firing rates index
                db.create_col('syllable_pcc', 'cvFRUndir', 'REAL')
                db.create_col('syllable_pcc', 'cvFRDir', 'REAL')
                db.conn.commit()

                for context in fr_cv.keys():
                    if ni.nb_note[context] >= nb_note_crit:
                        col_name = 'cvFRUndir' if context == 'U' else 'cvFRDir'
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET {col_name} = ({fr_cv[context]}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                # Add Fano Factor of firing rates index
                db.create_col('syllable_pcc', 'fanoFactorUndir', 'REAL')
                db.create_col('syllable_pcc', 'fanoFactorDir', 'REAL')
                db.conn.commit()

                for context, fano_factor in pi.fano_factor.items():
                    if ni.nb_note[context] >= nb_note_crit:
                        col_name = 'fanoFactorUndir' if context == 'U' else 'fanoFactorDir'
                        fano_factor = round(np.nanmean(fano_factor), 3)
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET {col_name} = ({fano_factor}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if shuffled_baseline:
                    if 'U' in p_sig and ni.nb_note['U'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET pccUndirSig = ({p_sig['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                    if 'D' in p_sig and ni.nb_note['D'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET pccDirSig = ({p_sig['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if entropy:
                    if 'U' in entropy_mean and ni.nb_note['U'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET entropyUndir = ({entropy_mean['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                    if 'D' in entropy_mean and ni.nb_note['D'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET entropyDir = ({entropy_mean['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                    if 'U' in spectro_temporal_entropy_mean and ni.nb_note['U'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET spectroTempEntropyUndir = ({spectro_temporal_entropy_mean['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                    if 'D' in spectro_temporal_entropy_mean and ni.nb_note['D'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET spectroTempEntropyDir = ({spectro_temporal_entropy_mean['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                    if 'U' in entropy_var and ni.nb_note['U'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET entropyVarUndir = ({entropy_var['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                    if 'D' in entropy_var and ni.nb_note['D'] >= nb_note_crit:
                        db.cur.execute(
                            f"UPDATE syllable_pcc SET entropyVarDir = ({entropy_var['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                db.conn.commit()

            # Save results
            if save_fig:
                save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
                save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=True)
            else:
                plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('syllable_pcc')
    print('Done!')


if __name__ == '__main__':

    # Parameter
    pre_buffer = 50  # time window before syllable onset (in ms)
    post_buffer = 0   # time window after syllable offset (in ms)
    time_warp = True  # spike time warping
    update = False  # set True for recreating a cache file
    update_db = False  # save results to DB
    entropy = False  # calculate entropy & entropy variance
    entropy_mode = 'spectral'  # computes time-resolved version of entropy ('spectral' or 'spectro_temporal')
    shuffled_baseline = False  # get pcc shuffling baseline
    plot_hist = False  # draw histogram of the shuffled pcc values
    fig_ext = '.png'  # .png or .pdf
    save_fig = True
    save_folder_name = 'RasterSyllable'
    TARGET_NOTE = 'all'  # notes to plot (set to 'all' to plot all syllables)
    NOTE_CONTEXT= 'U'  # context to plot ('U', 'D', set to None if you want to plot both)

    # SQL statement
    query = "SELECT * FROM cluster WHERE id=96"

    get_raster_syllable()
