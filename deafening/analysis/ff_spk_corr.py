"""
Get correlation between fundamental frequency and spike number
"""


def get_ff_spk_corr(query,
                    update=False,
                    save_spectrogram=False,
                    save_result_fig=False,
                    view_folder=False,
                    update_db=False,
                    save_csv=False,
                    fig_ext='.png'):

    from analysis.parameters import note_buffer, freq_range, nb_note_crit, pre_motor_win_size, alpha
    from analysis.spike import ClusterInfo, AudioData
    from analysis.functions import get_ff
    from util import save
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    from util.draw import remove_right_top
    import pandas as pd

    # Make save path
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'FF_SpkCorr', add_date=False)

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
        audio = AudioData(path, update=update)  # audio object

        # Fundamental Frequency analysis
        # Retrieve data from ff database


        db.execute(
            f"SELECT ffNote, ffParameter, ffCriterion, ffLow, ffHigh, ffDuration, harmonic FROM ff WHERE birdID='{cluster_db.birdID}'")
        ff_info = {data[0]: {'parameter': data[1],
                             'crit': data[2],
                             'low': data[3],  # lower limit of frequency
                             'high': data[4],  # upper limit of frequency
                             'duration': data[5],
                             'harmonic': data[6]  # 1st or 2nd harmonic detection
                             } for data in db.cur.fetchall()  # ff duration
                   }

        if not bool(ff_info):
            print("FF note doesn't exist")
            continue
        else:
            # Update ff_spk_corr db with note info
            for ff_note in ff_info.keys():
                query = f"INSERT OR IGNORE INTO ff_spk_corr (clusterID, birdID, taskName, taskSession, note) " \
                        f"VALUES({cluster_db.id}, '{cluster_db.birdID}', '{cluster_db.taskName}', {cluster_db.taskSession}, '{ff_note}')"
                db.cur.execute(query)
            db.conn.commit()

        # Load if cvs that has calculated FF and spike count already exists
        csv_name = ci.name + '.csv'
        csv_path = save_path / csv_name
        if csv_path.exists() and not save_spectrogram:
            df = pd.read_csv(csv_path)
        else:
            # Calculate FF
            df = pd.DataFrame()  # Store results here

            for note in set([note[0] for note in ff_info.keys()]):

                # Load note object
                ni = ci.get_note_info(note)
                if not ni or (
                        np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()])):  # if the note does not exist
                    db.cur.execute(
                        f"DELETE FROM ff_spk_corr WHERE clusterID= {cluster_db.id} AND note= '{note}'")  # delete the row from db
                    db.conn.commit()
                    continue

                print(f"Processing note {note}")
                # # Skip if there are not enough motifs per condition
                # if np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()]):
                #     # print("Not enough notes")
                #     continue

                zipped_lists = zip(ni.onsets, ni.offsets, ni.contexts, ni.spk_ts)
                for note_ind, (onset, offset, context, spk_ts) in enumerate(zipped_lists):

                    # Process only one particular note
                    # if note_ind != 6:
                    #     continue

                    # Loop through the notes
                    for i in range(0, [note[0] for note in ff_info.keys()].count(
                            note)):  # if more than one FF can be detected in a single note
                        # Note start and end
                        if [note[0] for note in ff_info.keys()].count(note) >= 2:
                            ff_note = f'{note}{i + 1}'
                        else:
                            ff_note = note

                        # if not ff_note == 'c':
                        #     continue

                        duration = offset - onset

                        # Get FF onset and offset based on the parameters from DB
                        if ff_info[ff_note]['crit'] == 'percent_from_start':
                            ff_onset = onset + (duration * (ff_info[ff_note]['parameter'] / 100))
                            ff_offset = ff_onset + ff_info[ff_note]['duration']
                        elif ff_info[ff_note]['crit'] == 'ms_from_start':
                            ff_onset = onset + (ff_info[ff_note]['parameter'])
                            ff_offset = ff_onset + ff_info[ff_note]['duration']
                        elif ff_info[ff_note]['crit'] == 'ms_from_end':
                            ff_onset = offset - ff_info[ff_note]['parameter']
                            ff_offset = ff_onset + ff_info[ff_note]['duration']

                        _, data = audio.extract([ff_onset, ff_offset])  # Extract audio data within the FF range

                        # Calculate fundamental frequency
                        ff = get_ff(data, audio.sample_rate,
                                    ff_info[ff_note]['low'], ff_info[ff_note]['high'],
                                    ff_harmonic=ff_info[ff_note]['harmonic'])

                        if not ff:  # skip the note if the ff is out of the expected range
                            continue

                        if save_spectrogram:
                            # Get spectrogram
                            # Load audio object with info from .not.mat files
                            timestamp, data = audio.extract([onset, offset])
                            spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)

                            # Plot figure
                            font_size = 8
                            fig = plt.figure(figsize=(4, 3), dpi=300)
                            fig_name = f"#{note_ind :03} - {ff_note} - {context}"

                            plt.suptitle(fig_name, y=.93, fontsize=font_size)
                            gs = gridspec.GridSpec(4, 6)

                            # Plot spectrogram
                            ax_spect = plt.subplot(gs[1:3, 0:4])
                            spect_time = spect_time - spect_time[0]  # starts from zero
                            ax_spect.pcolormesh(spect_time, spect_freq, spect,  # data
                                                cmap='hot_r',
                                                norm=colors.SymLogNorm(linthresh=0.05,
                                                                       linscale=0.03,
                                                                       vmin=0.5,
                                                                       vmax=100
                                                                       ))

                            remove_right_top(ax_spect)
                            ax_spect.set_xlim(-note_buffer, duration + note_buffer)
                            ax_spect.set_ylim(freq_range[0], freq_range[1])
                            ax_spect.set_xlabel('Time (ms)', fontsize=font_size)
                            ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
                            x_range = np.array(
                                [freq_range[0], 1000, 2000, 3000, 4000, 5000, 6000, 7000, freq_range[-1]])
                            plt.yticks(x_range, list(map(str, x_range)), fontsize=5)
                            plt.xticks(fontsize=5)

                            # Mark FF
                            ax_spect.axvline(x=ff_onset - onset, color='b', linewidth=0.5)
                            ax_spect.axvline(x=ff_offset - onset, color='b', linewidth=0.5)

                            # Mark estimated FF
                            ax_spect.axhline(y=ff, color='g', ls='--', lw=0.8)

                            # Print out text results
                            txt_xloc = -1.2
                            txt_yloc = 0.8
                            txt_offset = 0.2

                            ax_txt = plt.subplot(gs[1:, -1])
                            ax_txt.set_axis_off()  # remove all axes
                            ax_txt.text(txt_xloc, txt_yloc,
                                        f"{ff_info[ff_note]['parameter']} {ff_info[ff_note]['crit']}",
                                        fontsize=font_size)
                            txt_yloc -= txt_offset
                            ax_txt.text(txt_xloc, txt_yloc, f"ff duration = {ff_info[ff_note]['duration']} ms",
                                        fontsize=font_size)
                            txt_yloc -= txt_offset
                            ax_txt.text(txt_xloc, txt_yloc, f"ff = {ff} Hz", fontsize=font_size)

                            # Save figure
                            save_path2 = save.make_dir(save_path / ci.name, note, add_date=False)
                            save.save_fig(fig, save_path2, fig_name, view_folder=view_folder, fig_ext=fig_ext)

                        # Get number of spikes from pre-motor window
                        pre_motor_win = [ff_onset - pre_motor_win_size, ff_onset]
                        nb_spk = len(spk_ts[np.where((spk_ts >= pre_motor_win[0]) & (spk_ts <= pre_motor_win[-1]))])

                        # Organize results per song session
                        temp_df = pd.DataFrame({'note': [ff_note],
                                                'context': [context],
                                                'ff': [ff],
                                                'nb_spk': [nb_spk]
                                                })
                        df = df.append(temp_df, ignore_index=True)

            # Save ff and premotor spikes of each syllable rendition to csv
            if save_csv:
                df.index.name = 'Index'
                csv_name = ci.name + '.csv'
                df.to_csv(save_path / csv_name, index=True, header=True)  # save the dataframe to .cvs format

        # Draw correlation scatter between spk count and FF
        if not df.empty:
            font_size = 15
            for i, note in enumerate(sorted(df['note'].unique())):
                for context in sorted(df['context'].unique(), reverse=True):
                    temp_df = df[(df['note'] == note) & (df['context'] == context)]

                    # Go ahead with calculation only if it meets the number of notes criteria
                    if len(temp_df) >= nb_note_crit:

                        if 'fig2' not in locals():
                            fig2 = plt.figure(figsize=(len(set(df['note'])) * 6, len(set(df['context'].unique())) * 6))
                            plt.suptitle(ci.name, y=.9, fontsize=font_size)

                        # Calculate values
                        pre_motor_fr = round(
                            temp_df['nb_spk'].sum() / (len(temp_df['nb_spk']) * (pre_motor_win_size / 1E3)),
                            3)  # firing rates during the pre-motor window
                        corr, corr_pval = pearsonr(temp_df['nb_spk'], temp_df['ff'])
                        pval_sig = True if corr_pval < alpha else False
                        r_square = corr ** 2
                        polarity = 'positive' if corr > 0 else 'negative'

                        def get_shuffled_sig_prop():
                            shuffle_iter = 100
                            sig_array = np.array([], dtype=np.int)
                            for i in range(shuffle_iter):
                                shuffle_df = temp_df
                                corr, corr_pval = pearsonr(temp_df['nb_spk'], shuffle_df['ff'].sample(frac=1))
                                pval_sig = True if corr_pval < alpha else False
                                sig_array = np.append(sig_array, pval_sig)
                            return sig_array.mean()

                        shuffled_sig_prop = get_shuffled_sig_prop()

                        if len(df['context'].unique()) == 2 and context == 'U':
                            ax = plt.subplot2grid((13, len(set(df['note'])) + 1), (1, i), rowspan=3, colspan=1)
                        elif len(df['context'].unique()) == 2 and context == 'D':
                            ax = plt.subplot2grid((13, len(set(df['note'])) + 1), (8, i), rowspan=3, colspan=1)
                        else:
                            ax = plt.subplot2grid((7, len(set(df['note'])) + 1), (1, i), rowspan=3, colspan=1)

                        ax.scatter(temp_df['nb_spk'], temp_df['ff'], color='k', s=5)
                        ax.set_title(f"Note ({note}) - {context}", size=font_size)
                        if i == 0:
                            ax.set_ylabel('Note similarity')
                            ax.text(0, 0.5, context, fontsize=20)
                        ax.set_xlabel('Spk Count')
                        ax.set_xlim([-0.5, temp_df['nb_spk'].max() + 1])
                        remove_right_top(ax)

                        # Print out results
                        if len(df['context'].unique()) == 2 and context == 'U':
                            ax_txt = plt.subplot2grid((13, len(set(df['note'])) + 1), (4, i), rowspan=1, colspan=1)
                        elif len(df['context'].unique()) == 2 and context == 'D':
                            ax_txt = plt.subplot2grid((13, len(set(df['note'])) + 1), (11, i), rowspan=1, colspan=1)
                        else:
                            ax_txt = plt.subplot2grid((7, len(set(df['note'])) + 1), (4, i), rowspan=1, colspan=1)

                        txt_xloc = 0
                        txt_yloc = 0
                        txt_inc = 0.6

                        ax_txt.text(txt_xloc, txt_yloc, f"nbNotes = {len(temp_df)}", fontsize=font_size)
                        txt_yloc -= txt_inc
                        ax_txt.text(txt_xloc, txt_yloc, f"PremotorFR = {round(pre_motor_fr, 3)} (Hz)",
                                    fontsize=font_size)
                        txt_yloc -= txt_inc
                        ax_txt.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 3)}", fontsize=font_size)
                        txt_yloc -= txt_inc
                        t = ax_txt.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 3)}", fontsize=font_size)
                        if corr_pval < alpha:
                            corr_sig = True
                            t.set_bbox(dict(facecolor='green', alpha=0.5))
                        else:
                            corr_sig = False
                            t.set_bbox(dict(facecolor='red', alpha=0.5))

                        txt_yloc -= txt_inc
                        ax_txt.text(txt_xloc, txt_yloc, f"R_square = {round(r_square, 3)}", fontsize=font_size)
                        ax_txt.axis('off')

                    if update_db:
                        if context == 'U':
                            db.cur.execute(
                                f"UPDATE ff_spk_corr SET nbNoteUndir={len(temp_df)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            if len(temp_df) >= nb_note_crit:
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET ffMeanUndir={temp_df['ff'].mean() :1.3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET ffUndirCV={temp_df['ff'].std() / temp_df['ff'].mean() * 100 : .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET premotorFRUndir={pre_motor_fr} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrRUndir={round(corr, 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrPvalSigUndir={pval_sig} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET polarityUndir='{polarity}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrRsquareUndir='{round(r_square, 3)}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET shuffledSigPropUndir='{shuffled_sig_prop}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                        elif context == 'D':
                            db.cur.execute(
                                f"UPDATE ff_spk_corr SET nbNoteDir={len(temp_df)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            if len(temp_df) >= nb_note_crit:
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET ffMeanDir={temp_df['ff'].mean() :1.3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET ffDirCV={temp_df['ff'].std() / temp_df['ff'].mean() * 100 : .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET premotorFRDir={pre_motor_fr} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrRDir={round(corr, 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrPvalSigDir={pval_sig} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET polarityDir='{polarity}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET spkCorrRsquareDir='{round(r_square, 3)}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_spk_corr SET shuffledSigPropDir='{shuffled_sig_prop}' WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                        db.conn.commit()

                if update_db:
                    # If neither condition meets the number of notes criteria
                    db.cur.execute(
                        f"SELECT nbNoteUndir, nbNoteDir FROM ff_spk_corr WHERE clusterID={cluster_db.id} AND note= '{note}'")
                    nb_notes = [{'U': data[0], 'D': data[1]} for data in db.cur.fetchall()][0]
                    if not (bool(nb_notes['U']) or bool(nb_notes['D'])):
                        db.cur.execute(f"DELETE FROM ff_spk_corr WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                    db.conn.commit()

        if save_result_fig:
            save.save_fig(fig2, save_path, ci.name, view_folder=True, fig_ext=fig_ext)
        if 'fig2' in locals(): del fig2

    if update_db:
        db.to_csv('ff_spk_corr')

    print('Done!')


def get_sig_prop(df):
    """Get significant proportion of neurons & syllables"""
    import collections

    # Store number & proportion values here
    sig_prop = collections.defaultdict(dict)

    # Calculate sig neuronal proportion
    df_corr_sig = df.groupby(['clusterID', 'taskName'])['spkCorrPvalSigUndir'].sum().reset_index()
    df_shuffled_sig = df.groupby(['clusterID', 'taskName'])['shuffledSigPropUndir'].mean().reset_index()
    tasks = sorted(set(df['taskName']), reverse=True)
    for ind, task in enumerate(tasks):
        sig_array = df_corr_sig[(df_corr_sig['taskName'] == task)]['spkCorrPvalSigUndir']
        shuffled_sig_array = df_shuffled_sig[(df_shuffled_sig['taskName'] == task)]['shuffledSigPropUndir']
        sig_prop['baseline_neuron_prop'][task] = shuffled_sig_array.mean() + (2 * shuffled_sig_array.std())
        sig_prop['total_neurons'][task] = len(sig_array)
        sig_prop['sig_neurons'][task] = (sig_array >= 1).sum()
        sig_prop['sig_neuron_prop'][task] = (sig_array >= 1).sum() / len(sig_array)

    # Calculate sig syllable proportions
    for ind, task in enumerate(tasks):
        sig_array = df[(df['taskName'] == task)]['spkCorrPvalSigUndir']
        shuffled_sig_array = df[(df['taskName'] == task)]['shuffledSigPropUndir']
        sig_prop['baseline_syllable_prop'][task] = shuffled_sig_array.mean() + (2 * shuffled_sig_array.std())
        sig_prop['total_syllables'][task] = len(sig_array)
        sig_prop['sig_syllables'][task] = (sig_array >= 1).sum()
        sig_prop['sig_syllable_prop'][task] = (sig_array >= 1).sum() / len(sig_array)

    return sig_prop


def draw_sig_prop(df, sig_prop,
                  title, y_lim=None, save_fig=True,
                  fig_name=None, fig_ext='.png'):
    """Plot significant proportions of neurons / FF syllables"""
    from util.stats import z_test
    from util.draw import remove_right_top
    from scipy.stats import fisher_exact

    tasks = sorted(set(df['taskName']), reverse=True)
    # sig_neurons, total_neurons, baseline_neuron_prop, sig_neuron_prop,
    # sig_syllables, total_syllables, baseline_syllable_prop, sig_syllable_prop,

    # Plot the results
    fig = plt.figure(figsize=(7, 5))
    plt.suptitle(title, y=.9, fontsize=15)
    ax = plt.subplot2grid((6, 5), (1, 0), rowspan=3, colspan=2)

    for ind, task in enumerate(tasks):

        ax.bar(ind, sig_prop['sig_neuron_prop'][task], color='k')
        ax.set_ylabel('% of sig neurons')
        if y_lim:
            ax.set_ylim(y_lim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(tasks)
        ax.text(ind - 0.20, ax.get_ylim()[-1], f"({sig_prop['sig_neurons'][task]} / {sig_prop['total_neurons'][task]})",
                c='k')
        # Mark baseline proportion
        if ind == 0:
            ax.axhline(y=sig_prop['baseline_neuron_prop'][task],
                       xmin=ind + 0.05, xmax=ind + 0.45,
                       color='r', ls='--', lw=1)
        else:
            ax.axhline(y=sig_prop['baseline_neuron_prop'][task],
                       xmin=ind - 0.05, xmax=ind - 0.45,
                       color='r', ls='--', lw=1)
        remove_right_top(ax)

    ax = plt.subplot2grid((6, 5), (1, 3), rowspan=3, colspan=2)

    for ind, task in enumerate(tasks):

        ax.bar(ind, sig_prop['sig_syllable_prop'][task], color='k')
        ax.set_ylabel('% of sig syllables')
        if y_lim:
            ax.set_ylim(y_lim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(tasks)
        ax.text(ind - 0.2, ax.get_ylim()[-1],
                f"({sig_prop['sig_syllables'][task]} / {sig_prop['total_syllables'][task]})", c='k')
        # Mark baseline proportion
        if ind == 0:
            ax.axhline(y=sig_prop['baseline_syllable_prop'][task],
                       xmin=ind + 0.05, xmax=ind + 0.45,
                       color='r', ls='--', lw=1)
        else:
            ax.axhline(y=sig_prop['baseline_syllable_prop'][task],
                       xmin=ind - 0.05, xmax=ind - 0.45,
                       color='r', ls='--', lw=1)
        remove_right_top(ax)

    # Print out results
    # Proportion Z-test
    ax_txt = plt.subplot2grid((6, 5), (4, 0), rowspan=1, colspan=1)

    stat, pval_z = z_test(sig_prop['sig_neurons']['Predeafening'], sig_prop['total_neurons']['Predeafening']
                          , sig_prop['sig_neurons']['Postdeafening'], sig_prop['total_neurons']['Postdeafening'])

    odds_ratio, pval_fisher = fisher_exact(
        [[sig_prop['sig_neurons']['Predeafening'], sig_prop['total_neurons']['Predeafening']]
            , [sig_prop['sig_neurons']['Postdeafening'], sig_prop['total_neurons']['Postdeafening']]])

    font_size = 9
    txt_xloc = 0
    txt_yloc = 0.2
    txt_inc = 0.5

    ax_txt.set_ylim([0, 1])
    ax_txt.text(txt_xloc, txt_yloc, f"Z = {round(stat, 3)}", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"p_val (z-test) = {round(pval_z, 3)}", fontsize=font_size)
    txt_yloc -= txt_inc

    ax_txt.text(txt_xloc, txt_yloc, "Fisher's exact test", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"Odds ratio = {round(odds_ratio, 3)} ", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"p_val (Fisher's) = {round(pval_fisher, 3)} ", fontsize=font_size)
    ax_txt.axis('off')

    # Fisher's exact test
    ax_txt = plt.subplot2grid((6, 5), (4, 3), rowspan=1, colspan=1)

    stat, pval_z = z_test(sig_prop['sig_syllables']['Predeafening'], sig_prop['total_syllables']['Predeafening']
                          , sig_prop['sig_syllables']['Postdeafening'], sig_prop['total_syllables']['Postdeafening'])

    odds_ratio, pval = fisher_exact(
        [[sig_prop['sig_syllables']['Predeafening'], sig_prop['total_syllables']['Predeafening']]
            , [sig_prop['sig_syllables']['Postdeafening'], sig_prop['total_syllables']['Postdeafening']]])

    font_size = 9
    txt_xloc = 0.2
    txt_yloc = 0.2

    ax_txt.set_ylim([0, 1])
    ax_txt.text(txt_xloc, txt_yloc, f"Z = {round(stat, 3)}", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"p_val (z-test) = {round(pval_z, 3)}", fontsize=font_size)
    txt_yloc -= txt_inc

    ax_txt.text(txt_xloc, txt_yloc, "Fisher's exact test", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"Odds ratio = {round(odds_ratio, 3)} ", fontsize=font_size)
    txt_yloc -= txt_inc
    ax_txt.text(txt_xloc, txt_yloc, f"p_val (Fisher's) = {round(pval, 3)} ", fontsize=font_size)
    ax_txt.axis('off')
    plt.tight_layout()

    if save_fig:
        save.save_fig(fig, save_path, fig_name, view_folder=True, fig_ext=fig_ext)
    else:
        plt.show()


if __name__ == '__main__':

    from database.load import create_db, DBInfo, ProjectLoader
    import matplotlib.pyplot as plt
    import pandas as pd
    from util import save

    # Parameter
    save_spectrogram = False
    save_result_fig = True  # correlation figure
    view_folder = False  # view the folder where figures are stored
    update_db = False  # save results to DB
    save_csv = True  # save results to csv per neuron
    fig_ext = '.png'  # .png or .pdf
    nb_note_crit = 10
    fr_crit = 10

    # Create & Load database
    if update_db:
        db = create_db('create_ff_spk_corr.sql')

    # SQL statement
    # query = "SELECT * FROM cluster WHERE analysisOK"
    query = "SELECT * FROM cluster WHERE id=90"

    # Make save path
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'FF_SpkCorr', add_date=False)

    # Calculate results
    # get_ff_spk_corr(query,
    #                 save_spectrogram=save_spectrogram,
    #                 save_result_fig=save_result_fig,
    #                 view_folder=view_folder,
    #                 update_db=update_db,
    #                 save_csv=save_csv,
    #                 fig_ext=fig_ext)

    csv_path = ProjectLoader().path / 'Analysis/Database/ff_spk_corr.csv'
    df = pd.read_csv(csv_path, index_col='id')
    df = df.query(f"nbNoteUndir >= {nb_note_crit} and premotorFRUndir >= {fr_crit}")

    sig_prop = get_sig_prop(df)

    # Get proportion of neurons showing significant correlation per task
    title = f"{'Fundamental Frequency'} FR >= {fr_crit} (Undir)"

    draw_sig_prop(df, sig_prop,
                  title, y_lim=[0, 0.2], save_fig=True,
                  fig_name='ff_spk_corr')
