"""
By Jaerong
plot raster & peth
"""

def create_db():
    from database.load import ProjectLoader

    db = ProjectLoader().load_db()
    with open('database/create_pcc.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

def get_raster(query,
               shuffled_baseline = False,
               norm_method = None,
               fig_ext = '.png',
               time_warp = True,
               update = False,
               save_fig = True,
               update_db = True):
    """
    Plot raster & peri-event histogram

    Parameters
    ----------
    query : str
        SQL selection statement from cluster database
    shuffled_baseline : bool
        Get PETH from shuffled spikes for getting pcc baseline
    norm_method : bool
        Set True to normalize firing rates
    fig_ext : str
        Figure extension ('.png' or '.pdf' for vectorized figure)
    time_warp : bool
        Set True to perform piecewise linear warping
    update : bool
        Update cluster cache file
    save_fig : bool
        Set True to save figure
    update_db : bool
        Set True to update results to database
    """

    from analysis.parameters import peth_parm, freq_range, tick_length, tick_width, note_color, nb_note_crit, peth_shuffle
    from analysis.spike import MotifInfo, AudioData
    from collections import defaultdict
    from database.load import DBInfo, ProjectLoader
    from functools import partial
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from util import save
    from util.functions import myround
    from util.draw import remove_right_top

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
        audio = AudioData(path, update=update).extract([start, end])  # audio object
        audio.spectrogram(freq_range=freq_range)

        # Plot figure
        fig = plt.figure(figsize=(8, 9), dpi=500)

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
        line_offsets = np.arange(0.5, len(mi))
        if time_warp:
            zipped_lists = zip(mi.contexts, mi.spk_ts_warp, mi.onsets)
        else:
            zipped_lists = zip(mi.contexts, mi.spk_ts, mi.onsets)
        ax_raster = plt.subplot(gs[4:6, 0:4], sharex=ax_spect)

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
                                              alpha=0.15,
                                              facecolor=note_color['Motif'][i])
                elif not i % 2:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((sum(note_duration[:i]), motif_ind), note_duration[i], rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15,
                                              facecolor=note_color['Motif'][i - k])
                    k += 1
                ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=motif_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, (motif_ind))
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
        line_offsets = np.arange(0.5, len(mi))

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
                                              alpha=0.15,
                                              facecolor=note_color['Motif'][i])
                elif not i % 2:
                    # print("i is {}, color is {}".format(i, i-k))
                    rectangle = plt.Rectangle((sum(note_duration[:i]), motif_ind), note_duration[i], rec_height,
                                              fill=True,
                                              linewidth=1,
                                              alpha=0.15,
                                              facecolor=note_color['Motif'][i - k])
                    k += 1
                ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:

                ax_raster.axhline(y=motif_ind, color='k', ls='-', lw=0.3)
                context_change = np.append(context_change, (motif_ind))
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
        pi = mi.get_peth(time_warp=time_warp)  # peth object
        # pi.get_fr(norm_method='sum')  # get firing rates
        pi.get_fr(norm_method=norm_method)  # get firing rates

        ax_peth = plt.subplot(gs[10:12, 0:4], sharex=ax_spect)
        for context, mean_fr in pi.mean_fr.items():
            if context == 'U' and nb_motifs['U'] >= nb_note_crit:
                ax_peth.plot(pi.time_bin, mean_fr, 'b', label=context)
            elif context == 'D' and nb_motifs['D'] >= nb_note_crit:
                ax_peth.plot(pi.time_bin, mean_fr, 'm', label=context)

        plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5), prop={'size': 6})  # print out legend

        if norm_method:  # Normalize FR
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
            ax_txt.text(txt_xloc, txt_yloc, f"PCC (U) = {pi.pcc['U']['mean']}", fontsize=font_size)
        txt_yloc -= txt_inc

        if "D" in pi.pcc and nb_motifs['D'] >= nb_note_crit:
            ax_txt.text(txt_xloc, txt_yloc, f"PCC (D) = {pi.pcc['D']['mean']}", fontsize=font_size)



        # Get shuffled PETH
        if shuffled_baseline:

            pcc_shuffle = defaultdict(partial(np.ndarray, 0))
            for iter in range(peth_shuffle['shuffle_iter']):
                mi.jitter_spk_ts(peth_shuffle['shuffle_limit'])
                pi_shuffle = mi.get_peth(shuffle=True)  # peth object
                pi_shuffle.get_fr(norm_method=norm_method)  # get firing rates
                pi_shuffle.get_pcc()  # get pcc
                for context, pcc in pi_shuffle.pcc.items():
                    # pcc_shuffle[context].append(pcc['mean'])
                    pcc_shuffle[context] = np.append(pcc_shuffle[context], pcc['mean'])

        # One-sample t-test (one-sided)
        import scipy.stats as stats

        p_val = {}
        p_sig = {}
        alpha = 0.05

        for context in pcc_shuffle.keys():
            (_, p_val[context]) = stats.ttest_1samp(a=pcc_shuffle[context], popmean=pi.pcc[context]['mean'],
                                                    nan_policy='omit', alternative='less')
        for context, value in p_val.items():
            p_sig[context] = value < alpha

        # Plot histogram
        from util.draw import remove_right_top

        fig, axes = plt.subplots(1,2, figsize=(6, 3))
        plt.suptitle('PCC shuffle distribution', y=.98, fontsize=10)
        for axis, context in zip(axes, pcc_shuffle.keys()):
            axis.set_title(context)
            axis.hist(pcc_shuffle[context], color='k')
            axis.set_xlim([-0.1, 0.6])
            axis.set_xlabel('PCC'), axis.set_ylabel('Count')
            if p_sig[context]:
                axis.axvline(x=pi.pcc[context]['mean'], color='r', linewidth=1, ls='--')
            else:
                axis.axvline(x=pi.pcc[context]['mean'], color='k', linewidth=1, ls='--')
            remove_right_top(axis)
        plt.tight_layout()

        plt.show()

        # Corr context (correlation of firing rates between two contexts)
        txt_yloc -= txt_offset
        corr_context = np.nan
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
        for i, (context, cv) in enumerate(pi.spk_count_cv.items()):
            txt_yloc -= txt_inc
            if nb_motifs[context] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"CV of spk count ({context}) = {cv}", fontsize=font_size)

        # Plot fano factor
        ax_ff = plt.subplot(gs[16:18, 0:4], sharex=ax_spect)
        for context, fano_factor in pi.fano_factor.items():
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
        for i, (context, ff) in enumerate(pi.fano_factor.items()):
            txt_yloc -= txt_inc
            if nb_motifs[context] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"Fano Factor ({context}) = {round(np.nanmean(ff), 3)}", fontsize=font_size)

        # Save results to database
        if update_db and time_warp:  # only use values from time-warped data
            db.create_col('cluster', 'pairwiseCorrUndir', 'REAL')
            if 'U' in pi.pcc and nb_motifs['U'] >= nb_note_crit:
                db.update('cluster', 'pairwiseCorrUndir', row['id'], pi.pcc['U']['mean'])

            db.create_col('cluster', 'pairwiseCorrDir', 'REAL')
            if 'D' in pi.pcc and nb_motifs['D'] >= nb_note_crit:
                db.update('cluster', 'pairwiseCorrDir', row['id'], pi.pcc['D']['mean'])

            db.create_col('cluster', 'corrRContext', 'REAL')
            db.update('cluster', 'corrRContext', row['id'], corr_context)

            db.create_col('cluster', 'cvSpkCountUndir', 'REAL')
            if 'U' in pi.spk_count_cv and nb_motifs['U'] >= nb_note_crit:
                db.update('cluster', 'cvSpkCountUndir', row['id'], pi.spk_count_cv['U'])

            db.create_col('cluster', 'cvSpkCountDir', 'REAL')
            if 'D' in pi.spk_count_cv and nb_motifs['D'] >= nb_note_crit:
                db.update('cluster', 'cvSpkCountDir', row['id'], pi.spk_count_cv['D'])

            db.create_col('cluster', 'fanoSpkCountUndir', 'REAL')
            if 'U' in pi.fano_factor and nb_motifs['U'] >= nb_note_crit:
                db.update('cluster', 'fanoSpkCountUndir', row['id'], round(np.nanmean(pi.fano_factor['U']), 3))

            db.create_col('cluster', 'fanoSpkCountDir', 'REAL')
            if 'D' in pi.fano_factor and nb_motifs['D'] >= nb_note_crit:
                db.update('cluster', 'fanoSpkCountDir', row['id'], round(np.nanmean(pi.fano_factor['D']), 3))

            # if shuffled_baseline:
            #     db.cur.execute(f"UPDATE unit_profile SET burstDurationBaseline = ({burst_info_b.mean_duration}) WHERE clusterID = ({cluster_db.id})")

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Spk')
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
        else:
            plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('cluster')
    print('Done!')


if __name__ == '__main__':

    # Parameters
    shuffled_baseline = True
    fig_ext = '.png'
    time_warp = True
    update = False  # update the cache file
    save_fig = True
    update_db = False

    # Select from cluster db
    # query = "SELECT * FROM cluster WHERE analysisOK = 1"
    query = "SELECT * FROM cluster WHERE id = 96"

    # Create & Load database
    if update_db:
        db = create_db()

    get_raster(query, shuffled_baseline, fig_ext=fig_ext, time_warp=time_warp,
                   update=update,
                   save_fig=save_fig,
                   update_db=update_db)

