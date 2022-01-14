"""
plot raster & peth per syllable
calculate PCC across different time windows
"""


def get_raster_syllable(query,
                        buffer_size,
                        target_note=None,
                        update=False,
                        save_fig=None,
                        update_db=None,
                        time_warp=True,
                        fig_ext='.png'):

    from analysis.parameters import freq_range, peth_parm, note_color, tick_width, tick_length
    from analysis.spike import ClusterInfo, AudioData
    from database.load import create_db, DBInfo, ProjectLoader
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    from matplotlib import pyplot as plt
    import numpy as np
    from util import save
    from util.draw import remove_right_top
    from util.functions import find_str, myround
    import warnings
    warnings.filterwarnings('ignore')

    # parameters
    rec_yloc = 0.05
    rec_height = 1  # syllable duration rect
    text_yloc = 0.5  # text height
    font_size = 12
    marker_size = 0.4  # for spike count
    nb_note_crit = 10  # minimum number of notes for analysis

    # Create & Load database
    if update_db:
        db = create_db('create_syllable_pcc_window.sql')

    # Load database
    # SQL statement
    # Create a new database (syllable)
    db = ProjectLoader().load_db()
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        unit_nb = int(cluster_db.unit[-2:])
        channel_nb = int(cluster_db.channel[-2:])
        format = cluster_db.format

        # Load class object
        ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
        audio = AudioData(path, update=update)  # audio object

        # Loop through note
        for note in cluster_db.songNote:

            if target_note:
                if note is not target_note:
                    continue

            # Load note object
            ni = ci.get_note_info(note)
            if not ni:  # the target note does not exist
                continue

            # Skip if there are not enough motifs per condition
            if np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()]):
                print("Not enough notes")
                continue

            # Plot spectrogram & peri-event histogram (Just the first rendition)
            # Note start and end
            start = ni.onsets[0] - peth_parm['buffer']
            end = ni.offsets[0] + peth_parm['buffer']
            duration = ni.durations[0]

            # Get spectrogram
            # Load audio object with info from .not.mat files
            timestamp, data = audio.extract([start, end])
            spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)

            # Plot figure
            fig = plt.figure(figsize=(7, 10))
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
            spect_time = spect_time - spect_time[0] - peth_parm['buffer']  # starts from zero
            ax_spect.pcolormesh(spect_time, spect_freq, spect,  # data
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
            plt.setp(ax_raster.get_xticklabels(), visible=False)
            remove_right_top(ax_raster)

            # Plot sorted raster
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
            # zipped_lists = zip(ni.contexts, ni.spk_ts, ni.onsets)

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
            pi = ni.get_note_peth()  # PETH object (PethInfo)
            pi.get_fr()  # get firing rates

            # Plot mean firing rates
            ax_peth = plt.subplot(gs[10:12, 0:5], sharex=ax_spect)
            for context, fr in pi.mean_fr.items():
                if context == 'U':
                    ax_peth.plot(pi.time_bin, fr, 'b', label=context)
                elif context == 'D':
                    ax_peth.plot(pi.time_bin, fr, 'm', label=context)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})  # print out legend
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

            # Mark windows
            # Pre
            # rectangle = plt.Rectangle((-buffer_size, 0), ni.median_dur, ax_peth.get_ylim()[-1],
            #                           linewidth=1, alpha=0.1, edgecolor='k',
            #                           facecolor='k'
            #                           )
            # ax_peth.add_patch(rectangle)

            # Syllable
            # rectangle = plt.Rectangle((0, 0), ni.median_dur, ax_peth.get_ylim()[-1],
            #                           linewidth=1, alpha=0.1, edgecolor='k',
            #                           facecolor='k'
            #                           )
            # ax_peth.add_patch(rectangle)

            # Post
            # rectangle = plt.Rectangle((buffer_size, 0), ni.median_dur, ax_peth.get_ylim()[-1],
            #                           linewidth=1, alpha=0.1, edgecolor='k',
            #                           facecolor='k'
            #                           )
            # ax_peth.add_patch(rectangle)

            # Calculate pairwise cross-correlation across different time_windows
            pi_pre = ni.get_note_peth(pre_evt_buffer=buffer_size, duration=ni.median_dur - buffer_size)  # pre-motor window
            pi_pre.get_fr()
            pi_pre.get_pcc()

            pi_syllable = ni.get_note_peth(pre_evt_buffer=0, duration=ni.median_dur)  # pre-motor window
            pi_syllable.get_fr()
            pi_syllable.get_pcc()

            pi_post = ni.get_note_peth(pre_evt_buffer=-buffer_size, duration=ni.median_dur + buffer_size)  # pre-motor window
            pi_post.get_fr()
            pi_post.get_pcc()

            # Print out results on the figure
            txt_xloc = -2
            txt_yloc = 0.8
            txt_inc = 0.2  # y-distance between texts within the same section

            ax_txt = plt.subplot(gs[13:, 2])
            ax_txt.set_axis_off()  # remove all axes

            # # of notes
            #for i, (k, v) in enumerate(ni.nb_note.items()):
            #    ax_txt.text(txt_xloc, txt_yloc, f"# of notes ({k}) = {v}", fontsize=font_size)
            #    txt_yloc -= txt_inc

            # Print out firing rates
            if "U" in pi_pre.mean_fr.keys() and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR pre (U) = {pi_pre.mean_fr['U'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_pre.mean_fr.keys() and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR pre (D) = {pi_pre.mean_fr['D'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "U" in pi_syllable.mean_fr.keys() and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR syllable (U) = {pi_syllable.mean_fr['U'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_syllable.mean_fr.keys() and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR syllable (D) = {pi_syllable.mean_fr['D'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "U" in pi_post.mean_fr.keys() and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR post (U) = {pi_post.mean_fr['U'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_post.mean_fr.keys() and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"FR post (D) = {pi_post.mean_fr['D'].mean(): .3f}", fontsize=font_size)
            txt_yloc -= txt_inc

            # PCC (pre)
            txt_xloc = 1
            txt_yloc = 0.8

            if "U" in pi_pre.pcc and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC pre (U) = {pi_pre.pcc['U']['mean']}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_pre.pcc and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC pre (D) = {pi_pre.pcc['D']['mean']}", fontsize=font_size)
            txt_yloc -= txt_inc

            # PCC (syllable)
            if "U" in pi_syllable.pcc and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC syllable (U) = {pi_syllable.pcc['U']['mean']}",
                            fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_syllable.pcc and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC syllable (D) = {pi_syllable.pcc['D']['mean']}",
                            fontsize=font_size)
            txt_yloc -= txt_inc

            # PCC (post)
            if "U" in pi_post.pcc and ni.nb_note['U'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC post (U) = {pi_post.pcc['U']['mean']}", fontsize=font_size)
            txt_yloc -= txt_inc

            if "D" in pi_post.pcc and ni.nb_note['D'] >= nb_note_crit:
                ax_txt.text(txt_xloc, txt_yloc, f"PCC post (D) = {pi_post.pcc['D']['mean']}", fontsize=font_size)
            txt_yloc -= txt_inc

            #plt.show()

            # Save results to database
            if update_db:  # only use values from time-warped data
                sql = "INSERT OR IGNORE INTO " \
                        "syllable_pcc_window (clusterID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days, note)" \
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
                        f"UPDATE syllable_pcc_window SET nbNoteUndir = ({ni.nb_note['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in ni.nb_note:
                    db.cur.execute(
                        f"UPDATE syllable_pcc_window SET nbNoteDir = ({ni.nb_note['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'U' in pi_pre.mean_fr and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc_window "
                        f"SET frUndirPre = ({pi_pre.mean_fr['U'].mean(): .3f}),"
                        f"frUndirSyllable = ({pi_syllable.mean_fr['U'].mean(): .3f}),"
                        f"frUndirPost = ({pi_post.mean_fr['U'].mean(): .3f}) "
                        f"WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in pi_pre.mean_fr and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(
                        f"UPDATE syllable_pcc_window "
                        f"SET frDirPre = ({pi_pre.mean_fr['D'].mean(): .3f}),"
                        f"frDirSyllable = ({pi_syllable.mean_fr['D'].mean(): .3f}),"
                        f"frDirPost = ({pi_post.mean_fr['D'].mean(): .3f}) "
                        f"WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                # Undir
                if 'U' in pi_pre.pcc and not np.isnan(pi_pre.pcc['U']['mean']) and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccUndirPre = ({pi_pre.pcc['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'U' in pi_syllable.pcc and not np.isnan(pi_syllable.pcc['U']['mean']) and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccUndirSyllable = ({pi_syllable.pcc['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'U' in pi_post.pcc and not np.isnan(pi_post.pcc['U']['mean']) and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccUndirPost = ({pi_post.pcc['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                # Dir
                if 'D' in pi_pre.pcc and not np.isnan(pi_pre.pcc['D']['mean']) and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccDirPre = ({pi_pre.pcc['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in pi_syllable.pcc and not np.isnan(pi_syllable.pcc['D']['mean']) and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccDirSyllable = ({pi_syllable.pcc['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

                if 'D' in pi_post.pcc and not np.isnan(pi_post.pcc['U']['mean']) and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_window SET pccDirPost = ({pi_post.pcc['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                db.conn.commit()

            # Save results
            if save_fig:
                save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'RasterSyllable')
                save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=True)
            else:
                plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('syllable_pcc_window')
    print('Done!')


if __name__ == '__main__':

    # Parameter
    update = False  # Set True for recreating a cache file
    save_fig = True
    update_db = True  # save results to DB
    fig_ext = '.png'  # .png or .pdf
    target_note = None  # None if you want to plot all notes

    # SQL statement
    query = "SELECT * FROM cluster WHERE analysisOK"

    get_raster_syllable(query,
                        buffer_size=40,  # in ms
                        target_note=target_note,
                        save_fig=save_fig,
                        update_db=update_db,
                        fig_ext=fig_ext)
