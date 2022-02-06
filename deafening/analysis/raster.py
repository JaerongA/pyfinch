"""
Plot raster & peth for motif
Calculate spike temporal precision metrics such as pcc
Store results in pcc table
"""
import matplotlib.pyplot as plt
import numpy as np


def pcc_shuffle_test(MotifInfo, PethInfo, plot_hist=False):
    """
    Test whether pcc value is significant compared to the baseline pcc computed using shuffled spikes
    Parameters
    ----------
    MotifInfo : class object
    PethInfo : class objet
    plot_hist : bool
        plot pcc histogram if True

    Returns
    -------
    p_sig : dict
        dictionary contains significance for difference contexts
    """
    from analysis.parameters import peth_shuffle
    from collections import defaultdict
    from functools import partial
    import scipy.stats as stats

    pcc_shuffle = defaultdict(partial(np.ndarray, 0))
    for _ in range(peth_shuffle['shuffle_iter']):
        MotifInfo.jitter_spk_ts(peth_shuffle['shuffle_limit'])
        pi_shuffle = MotifInfo.get_note_peth(shuffle=True)  # peth object
        pi_shuffle.get_fr()  # get firing rates
        pi_shuffle.get_pcc()  # get pcc
        for context, pcc in pi_shuffle.pcc.items():
            pcc_shuffle[context] = np.append(pcc_shuffle[context], pcc['mean'])

    # One-sample t-test (one-sided)
    p_val = {}
    p_sig = {}
    alpha = 0.05

    for context in pcc_shuffle.keys():
        (_, p_val[context]) = stats.ttest_1samp(a=pcc_shuffle[context], popmean=PethInfo.pcc[context]['mean'],
                                                nan_policy='omit', alternative='less')
    for context, value in p_val.items():
        p_sig[context] = value < alpha

    # Plot histogram
    if plot_hist:
        from util.draw import remove_right_top

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plt.suptitle('PCC shuffle distribution', y=.98, fontsize=10)
        for axis, context in zip(axes, pcc_shuffle.keys()):
            axis.set_title(context)
            axis.hist(pcc_shuffle[context], color='k')
            axis.set_xlim([-0.1, 0.6])
            axis.set_xlabel('PCC'), axis.set_ylabel('Count')
            if p_sig[context]:
                axis.axvline(x=PethInfo.pcc[context]['mean'], color='r', linewidth=1, ls='--')
            else:
                axis.axvline(x=PethInfo.pcc[context]['mean'], color='k', linewidth=1, ls='--')
            remove_right_top(axis)
        plt.tight_layout()
        plt.show()

    return p_sig


def main():
    from analysis.parameters import peth_parm, freq_range, tick_length, tick_width, note_color, nb_note_crit
    from analysis.spike import MotifInfo, AudioData
    from database.load import create_db, DBInfo, ProjectLoader
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    from util import save
    from util.functions import myround
    from util.draw import remove_right_top
    import warnings
    warnings.filterwarnings('ignore')

    # Create & Load database
    if update_db:
        create_db('create_unit_profile.sql')

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
        mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name, update=update)  # cluster object

        audio = AudioData(path, update=update)  # audio object

        # Get number of motifs
        nb_motifs = mi.nb_motifs(motif)
        nb_motifs.pop('All', None)

        # Skip if there are not enough motifs per condition
        # if nb_motifs['U'] < nb_note_crit and nb_motifs['D'] < nb_note_crit:
        #     print("Not enough motifs")
        #     continue

        # Plot spectrogram & peri-event histogram (Just the first rendition)
        # for onset, offset in zip(mi.onsets, mi.offsets):
        onsets = mi.onsets[0]
        offsets = mi.offsets[0]

        # Convert from string to array of floats
        onsets = np.asarray(list(map(float, onsets)))
        offsets = np.asarray(list(map(float, offsets)))

        # Motif start and end
        start = onsets[0] - peth_parm['buffer']
        end = offsets[-1] + peth_parm['buffer']
        duration = offsets[-1] - onsets[0]

        # Get spectrogram
        timestamp, data = audio.extract([start, end])
        spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)

        # Plot figure
        fig = plt.figure(figsize=(8, 9), dpi=600)

        fig.set_tight_layout(False)
        if time_warp:
            fig_name = mi.name + '  (time-warped)'
        else:
            fig_name = mi.name + '  (non-warped)'
        plt.suptitle(fig_name, y=.93)
        gs = gridspec.GridSpec(18, 6)
        gs.update(wspace=0.025, hspace=0.05)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3, 0:4])
        spect_time = spect_time - spect_time[0] - peth_parm['buffer']  # starts from zero
        ax_spect.pcolormesh(spect_time, spect_freq, spect,
                            cmap='hot_r', rasterized=True,
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        x_max = myround(duration + peth_parm['buffer'], base=100)
        ax_spect.set_xlim(-peth_parm['buffer'], x_max)
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0, 0:4], sharex=ax_spect)
        note_dur = offsets - onsets  # syllable duration
        onsets -= onsets[0]  # start from 0
        offsets = onsets + note_dur

        # Mark syllables
        for i, syl in enumerate(mi.motif):
            rectangle = plt.Rectangle((onsets[i], rec_yloc), note_dur[i], 0.2,
                                      linewidth=1, alpha=0.5, edgecolor='k', facecolor=note_color['Motif'][i])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onsets[i] + (offsets[i] - onsets[i]) / 2), text_yloc, syl, size=font_size)
        ax_syl.axis('off')

        # Plot raster
        ax_raster = plt.subplot(gs[4:6, 0:4], sharex=ax_spect)

        line_offsets = np.arange(0.5, len(mi))
        if time_warp:
            zipped_lists = zip(mi.contexts, mi.spk_ts_warp, mi.onsets)
        else:
            zipped_lists = zip(mi.contexts, mi.spk_ts, mi.onsets)

        pre_context = ''  # for marking  context change
        context_change = np.array([])

        for motif_ind, (context, spk_ts, onsets) in enumerate(zipped_lists):

            # Plot rasters
            spk = spk_ts - float(onsets[0])
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[motif_ind],
                                linelengths=tick_length, linewidths=tick_width, orientation='horizontal')

            # Demarcate the note
            if time_warp:
                note_duration = mi.median_durations
            else:  # plot (unwarped) raw data
                note_duration = mi.note_durations[motif_ind]

            k = 1  # index for setting the motif color
            for i, dur in enumerate(note_duration):

                if i == 0:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((0, motif_ind), dur, rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15, rasterized=True,
                                              facecolor=note_color['Motif'][i])
                elif not i % 2:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((sum(note_duration[:i]), motif_ind), note_duration[i], rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15, rasterized=True,
                                              facecolor=note_color['Motif'][i - k])
                    k += 1
                ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=motif_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, motif_ind)
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

        ax_raster.set_ylim(0, len(mi))
        ax_raster.set_ylabel('Trial #', fontsize=font_size)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        plt.yticks([0, len(mi)], [str(0), str(len(mi))])
        remove_right_top(ax_raster)

        # Plot sorted raster
        ax_raster = plt.subplot(gs[7:9, 0:4], sharex=ax_spect)

        # Sort trials based on context
        sort_ind = np.array([i[0] for i in sorted(enumerate(mi.contexts), key=lambda x: x[1], reverse=True)])
        mi.contexts_sorted = np.array(mi.contexts)[sort_ind].tolist()
        mi.onsets_sorted = np.array(mi.onsets)[sort_ind].tolist()
        if time_warp:
            mi.spk_ts_sorted = np.array(mi.spk_ts_warp)[sort_ind].tolist()
        else:
            mi.spk_ts_sorted = np.array(mi.spk_ts)[sort_ind].tolist()

        zipped_lists = zip(mi.contexts_sorted, mi.spk_ts_sorted, mi.onsets_sorted)

        pre_context = ''  # for marking  context change
        context_change = np.array([])

        for motif_ind, (context, spk_ts, onsets) in enumerate(zipped_lists):

            # Plot rasters
            spk = spk_ts - float(onsets[0])
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[motif_ind],
                                linelengths=tick_length, linewidths=tick_width, orientation='horizontal')

            # Demarcate the note
            if time_warp:
                note_duration = mi.median_durations
            else:  # plot (unwarped) raw data
                note_duration = mi.note_durations[motif_ind]

            k = 1  # index for setting the motif color
            for i, dur in enumerate(note_duration):

                if i == 0:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((0, motif_ind), dur, rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15, rasterized=True,
                                              facecolor=note_color['Motif'][i])
                elif not i % 2:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((sum(note_duration[:i]), motif_ind), note_duration[i], rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15, rasterized=True,
                                              facecolor=note_color['Motif'][i - k])
                    k += 1
                ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:

                ax_raster.axhline(y=motif_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, motif_ind)
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

        ax_raster.set_ylim(0, len(mi))
        ax_raster.set_ylabel('Trial #', fontsize=font_size)
        # ax_raster.set_xlabel('Time (ms)', fontsize=font_size)
        ax_raster.set_title('sorted raster', size=font_size)
        plt.yticks([0, len(mi)], [str(0), str(len(mi))])
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        remove_right_top(ax_raster)

        # Draw peri-event histogram (PETH)
        ax_peth = plt.subplot(gs[10:12, 0:4], sharex=ax_spect)

        pi = mi.get_peth(time_warp=time_warp)  # peth object
        # pi.get_fr(norm_method='sum')  # get firing rates
        pi.get_fr()  # get firing rates

        for context, mean_fr in pi.mean_fr.items():
            if context == 'U' and nb_motifs['U'] >= nb_note_crit:
                ax_peth.plot(pi.time_bin, mean_fr, 'b', label=context)
            elif context == 'D' and nb_motifs['D'] >= nb_note_crit:
                ax_peth.plot(pi.time_bin, mean_fr, 'm', label=context)

        plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend

        if normalize_fr:  # Normalize FR
            ax_peth.set_ylabel('Norm. FR', fontsize=font_size)
        else:  # Raw FR
            ax_peth.set_ylabel('FR', fontsize=font_size)

        fr_ymax = myround(round(ax_peth.get_ylim()[1], 3), base=5)
        ax_peth.set_ylim(0, fr_ymax)
        plt.yticks([0, ax_peth.get_ylim()[1]], [str(0), str(int(fr_ymax))])

        # Mark the baseline firing rates
        if 'baselineFR' in row.keys() and row['baselineFR']:
            ax_peth.axhline(y=row['baselineFR'], color='k', ls='--', lw=0.5)

        # Mark end of the motif
        ax_peth.axvline(x=0, color='k', ls='--', lw=0.5)
        ax_peth.axvline(x=mi.median_durations.sum(), color='k', lw=0.5)
        plt.setp(ax_peth.get_xticklabels(), visible=False)
        remove_right_top(ax_peth)

        # Calculate pairwise cross-correlation
        pi.get_pcc()

        # Get shuffled PETH & pcc
        if shuffled_baseline:
            p_sig = pcc_shuffle_test(mi, pi, plot_hist=True)

        # Calculate sparseness index
        sparseness = pi.get_sparseness(bin_size=3)

        # Print out results on the figure
        txt_xloc = -0.5
        txt_yloc = 1
        txt_inc = 0.05  # y-distance between texts within the same section
        txt_offset = 0.1

        ax_txt = plt.subplot(gs[::, 5])
        ax_txt.set_axis_off()  # remove all axes

        # # of motifs
        for i, (k, v) in enumerate(nb_motifs.items()):
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"# of motifs ({k}) = {v}", fontsize=font_size)

        # PCC
        txt_yloc -= txt_offset
        if "U" in pi.pcc and nb_motifs['U'] >= nb_note_crit:
            t = ax_txt.text(txt_xloc, txt_yloc, f"PCC (U) = {pi.pcc['U']['mean']}", fontsize=font_size)
            if "p_sig" in locals():
                if 'U' in p_sig and p_sig['U']:
                    t.set_bbox(dict(facecolor='green', alpha=0.5))
                else:
                    t.set_bbox(dict(facecolor='red', alpha=0.5))

        txt_yloc -= txt_inc

        if "D" in pi.pcc and nb_motifs['D'] >= nb_note_crit:
            t = ax_txt.text(txt_xloc, txt_yloc, f"PCC (D) = {pi.pcc['D']['mean']}", fontsize=font_size)
            if "p_sig" in locals():
                if 'D' in p_sig and p_sig['D']:
                    t.set_bbox(dict(facecolor='green', alpha=0.5))
                else:
                    t.set_bbox(dict(facecolor='red', alpha=0.5))

        # Corr context (correlation of firing rates between two contexts)
        txt_yloc -= txt_offset
        corr_context = None
        if 'U' in pi.mean_fr.keys() and 'D' in pi.mean_fr.keys() \
                and (nb_motifs['U'] >= nb_note_crit and nb_motifs['D'] >= nb_note_crit):
            corr_context = round(np.corrcoef(pi.mean_fr['U'], pi.mean_fr['D'])[0, 1], 3)
        ax_txt.text(txt_xloc, txt_yloc, f"Context Corr = {corr_context}", fontsize=font_size)

        # Plot spike counts
        pi.get_spk_count()  # spike count per time window
        ax_spk_count = plt.subplot(gs[13:15, 0:4], sharex=ax_spect)
        for context, spk_count in pi.spk_count.items():
            if context == 'U' and nb_motifs[context] >= nb_note_crit:
                ax_spk_count.plot(pi.time_bin, spk_count, 'o', color='b', mfc='none', linewidth=0.5, label=context,
                                  markersize=marker_size)
            elif context == 'D' and nb_motifs[context] >= nb_note_crit:
                ax_spk_count.plot(pi.time_bin, spk_count, 'o', color='m', mfc='none', linewidth=0.5, label=context,
                                  markersize=marker_size)

        plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend
        remove_right_top(ax_spk_count)
        ymax = myround(round(ax_spk_count.get_ylim()[1], 3), base=5)
        ax_spk_count.set_ylim(0, ymax)
        plt.yticks([0, ax_spk_count.get_ylim()[1]], [str(0), str(int(ymax))])
        ax_spk_count.set_ylabel('Spike Count', fontsize=font_size)
        ax_spk_count.axvline(x=0, color='k', ls='--', lw=0.5)
        ax_spk_count.axvline(x=mi.median_durations.sum(), color='k', ls='--', lw=0.5)
        plt.setp(ax_spk_count.get_xticklabels(), visible=False)

        # Print out results on the figure
        txt_yloc -= txt_inc
        for context, cv in sorted(pi.spk_count_cv.items(), reverse=True):
            txt_yloc -= txt_inc
            if nb_motifs[context] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"CV of spk count ({context}) = {cv}", fontsize=font_size)

        # Plot fano factor
        ax_ff = plt.subplot(gs[16:18, 0:4], sharex=ax_spect)
        for context, fano_factor in sorted(pi.fano_factor.items(), reverse=True):
            if context == 'U' and nb_motifs[context] >= nb_note_crit:
                ax_ff.plot(pi.time_bin, fano_factor, color='b', mfc='none', linewidth=0.5, label=context)
            elif context == 'D' and nb_motifs[context] >= nb_note_crit:
                ax_ff.plot(pi.time_bin, fano_factor, color='m', mfc='none', linewidth=0.5, label=context)

        plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend
        remove_right_top(ax_ff)
        ymax = round(ax_ff.get_ylim()[1], 2)
        ax_ff.set_ylim(0, ymax)
        plt.yticks([0, ax_ff.get_ylim()[1]], [str(0), str(int(ymax))])
        ax_ff.set_ylabel('Fano factor', fontsize=font_size)
        ax_ff.axvline(x=0, color='k', ls='--', lw=0.5)
        ax_ff.axvline(x=mi.median_durations.sum(), color='k', ls='--', lw=0.5)
        ax_ff.axhline(y=1, color='k', ls='--', lw=0.5)  # baseline for fano factor
        ax_ff.set_xlabel('Time (ms)', fontsize=font_size)

        # Print out results on the figure
        txt_yloc -= txt_inc
        for context, ff in sorted(pi.fano_factor.items(), reverse=True):
            txt_yloc -= txt_inc
            if nb_motifs[context] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"Fano Factor ({context}) = {round(np.nanmean(ff), 3)}",
                            fontsize=font_size)

        txt_yloc -= txt_inc
        for context, value in sorted(sparseness.items(), reverse=True):
            txt_yloc -= txt_inc
            if nb_motifs[context] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"Sparseness ({context}) = {value : 1.3f}",
                            fontsize=font_size)
        # Save results to database
        if update_db and time_warp:  # only use values from time-warped data
            db.cur.execute(f"INSERT OR IGNORE INTO pcc (clusterID) VALUES ({cluster_db.id})")
            db.conn.commit()

            if 'U' in pi.pcc and nb_motifs['U'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET pccUndir = ({pi.pcc['U']['mean']}) WHERE clusterID = ({cluster_db.id})")

            if 'D' in pi.pcc and nb_motifs['D'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET pccDir = ({pi.pcc['D']['mean']}) WHERE clusterID = ({cluster_db.id})")

            if corr_context:
                db.cur.execute(f"UPDATE pcc SET corrRContext = ({corr_context}) WHERE clusterID = ({cluster_db.id})")

            if 'U' in pi.spk_count_cv and nb_motifs['U'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET cvSpkCountUndir = ({pi.spk_count_cv['U']}) WHERE clusterID = ({cluster_db.id})")

            if 'D' in pi.spk_count_cv and nb_motifs['D'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET cvSpkCountDir = ({pi.spk_count_cv['D']}) WHERE clusterID = ({cluster_db.id})")

            if 'U' in pi.fano_factor and nb_motifs['U'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET fanoSpkCountUndir = ({round(np.nanmean(pi.fano_factor['U']), 3)}) WHERE clusterID = ({cluster_db.id})")

            if 'D' in pi.fano_factor and nb_motifs['D'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET fanoSpkCountDir = ({round(np.nanmean(pi.fano_factor['D']), 3)}) WHERE clusterID = ({cluster_db.id})")

            if 'U' in sparseness and nb_motifs['U'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET sparsenessUndir = ({sparseness['U'] :1.3f}) WHERE clusterID = ({cluster_db.id})")

            if 'D' in sparseness and nb_motifs['D'] >= nb_note_crit:
                db.cur.execute(
                    f"UPDATE pcc SET sparsenessDir = ({sparseness['D'] :1.3f}) WHERE clusterID = ({cluster_db.id})")

            if shuffled_baseline:
                if 'U' in p_sig and nb_motifs['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE pcc SET pccUndirSig = ({p_sig['U']}) WHERE clusterID = ({cluster_db.id})")
                if 'D' in p_sig and nb_motifs['D'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE pcc SET pccDirSig = ({p_sig['D']}) WHERE clusterID = ({cluster_db.id})")
            db.conn.commit()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder)
        else:
            plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('pcc')
    print('Done!')


if __name__ == '__main__':
    # Parameters
    time_warp = True  # perform piece-wise linear time-warping
    update = False  # Update the cache file (.npz) per cluster
    normalize_fr = True  # Set True to normalize firing rates
    shuffled_baseline = False  # get PETH from shuffled spikes for getting pcc baseline
    save_folder_name = 'Raster'  # Folder name to save figures
    save_fig = True  # Save the figure
    update_db = False  # update database
    view_folder = True  # open the folder where the result figures are saved
    fig_ext = '.pdf'  # set to '.pdf' for vector output (.png by default)
    NOTE_CONTEXT= 'U'  # context to plot ('U', 'D', set to None if you want to plot both)

    # SQL statement
    # Select from cluster table
    query = "SELECT * FROM cluster WHERE birdID='w16w14' AND analysisOK AND id>=33"
    # query = "SELECT * FROM cluster WHERE id=33"

    main()
