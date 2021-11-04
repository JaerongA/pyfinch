"""
By Jaerong
Calculate entropy, entropy variance per syllable
Stores the results in individual_syllable, syllable table
Figures saved in /Analysis/Entropy
"""


def get_entropy(query,
                update=False,
                nb_note_crit=None,
                save_fig=None,
                view_folder=False,
                update_db=False,
                fig_ext='.png'):

    from analysis.parameters import note_buffer, freq_range
    from analysis.song import AudioInfo, SongInfo
    from database.load import ProjectLoader, DBInfo
    from util import save
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from util.draw import remove_right_top
    import pandas as pd

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()
        song_note = song_db.songNote

        si = SongInfo(path, name, update=update)  # song object

        df = pd.DataFrame()  # Store results here

        note_ind1 = -1  # note index across the session

        for file in si.files:

            file = ProjectLoader().path / file
            # print(f'Loading... {file}')
            # Loop through the notes
            note_ind2 = -1  # note index within a file

            # Load audio object with info from .not.mat files
            ai = AudioInfo(file)
            ai.load_notmat()

            for note, onset, offset in zip(ai.syllables, ai.onsets, ai.offsets):

                if note not in song_db.songNote: continue  # skip if not a song note

                if update_db and note in song_note:
                    # Fill in song info
                    query = f"INSERT OR IGNORE INTO syllable (songID, birdID, taskName, note)" \
                            f"VALUES({song_db.id}, '{song_db.birdID}', '{song_db.taskName}', '{note}')"
                    db.cur.execute(query)
                    db.conn.commit()
                    song_note = song_note.replace(note, '')

                # Note start and end
                note_ind1 += 1  # note index across the session
                note_ind2 += 1  # note index within a file

                # if note_ind1 != 1:
                #     continue

                duration = offset - onset

                # Get spectrogram
                timestamp, data = ai.extract([onset, offset])  # Extract data within the range
                spect_time, spect, spect_freq = ai.spectrogram(timestamp, data)
                spectral_entropy = ai.get_spectral_entropy(spect, mode='spectral')
                se_dict = ai.get_spectral_entropy(spect, mode='spectro_temporal')

                if update_db and note in song_db.songNote:
                    query = f"INSERT OR IGNORE INTO individual_syllable (noteIndSession, noteIndFile, songID, fileID, birdID, taskName, note, context)" \
                            f"VALUES({note_ind1}, {note_ind2}, {song_db.id}, '{file.stem}', '{song_db.birdID}', '{song_db.taskName}', '{note}', '{ai.context}')"
                    db.cur.execute(query)
                    db.conn.commit()

                    query = f"UPDATE individual_syllable " \
                            f"SET entropy={round(spectral_entropy, 3)}, spectroTemporalEntropy={round(se_dict['mean'], 3)}, entropyVar={round(se_dict['var'], 4)} " \
                            f"WHERE noteIndSession={note_ind1} AND noteIndFile={note_ind2} AND songID={song_db.id}"
                    db.cur.execute(query)

                # Organize results per song session
                temp_df = pd.DataFrame({'note': [note], 'context': [ai.context],
                                        'spectral_entropy': [round(spectral_entropy, 3)],
                                        'spectro_temporal_entropy': [round(se_dict['mean'], 3)],
                                        'entropy_var': [round(se_dict['var'], 4)]
                                        })
                df = df.append(temp_df, ignore_index=True)

                if save_fig:
                    # Parameters
                    txt_offset = 0.2
                    font_size = 6

                    # Plot figure
                    fig = plt.figure(figsize=(4, 2), dpi=250)
                    fig_name = f"{note_ind1 :03} - {file.name}, note#{note_ind2} - {note}"

                    plt.suptitle(fig_name, y=.90, fontsize=font_size)
                    gs = gridspec.GridSpec(4, 6)

                    # Plot spectrogram
                    ax_spect = plt.subplot(gs[1:3, 0:3])
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
                    plt.yticks(freq_range, list(map(str, freq_range)), fontsize=5)
                    plt.xticks(fontsize=5), plt.yticks(fontsize=5)

                    # Calculate spectral entropy per time bin
                    # Plot syllable entropy
                    ax_se = ax_spect.twinx()
                    ax_se.plot(spect_time, se_dict['array'], 'k')
                    ax_se.set_ylim(0, 1)
                    ax_se.spines['top'].set_visible(False)
                    ax_se.set_ylabel('Entropy', fontsize=font_size)
                    plt.xticks(fontsize=5), plt.yticks(fontsize=5)

                    # Print out text results
                    txt_xloc = -1.5
                    txt_yloc = 0.8
                    ax_txt = plt.subplot(gs[1:, -1])
                    ax_txt.set_axis_off()  # remove all axes
                    ax_txt.text(txt_xloc, txt_yloc, f"Spectral Entropy = {round(spectral_entropy, 3)}",
                                fontsize=font_size)
                    txt_yloc -= txt_offset
                    ax_txt.text(txt_xloc, txt_yloc, f"Spectrotemporal Entropy = {round(se_dict['mean'], 3)}",
                                fontsize=font_size)
                    txt_yloc -= txt_offset
                    ax_txt.text(txt_xloc, txt_yloc, f"Entropy Variance = {round(se_dict['var'], 4)}",
                                fontsize=font_size)

                    # Save results
                    save_path2 = save.make_dir(save_path, si.name, add_date=False)
                    save.save_fig(fig, save_path2, fig_name, view_folder=view_folder, fig_ext=fig_ext)

        # Save results to ff_results db
        if not df.empty:
            if update_db:
                for note in df['note'].unique():
                    for context in df['context'].unique():
                        temp_df = df[(df['note'] == note) & (df['context'] == context)]
                        if context == 'U':
                            db.cur.execute(
                                f"UPDATE syllable SET nbNoteUndir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                            if len(temp_df) >= nb_note_crit:
                                db.cur.execute(
                                    f"UPDATE syllable SET entropyUndir={temp_df['spectral_entropy'].mean() : .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE syllable SET spectroTemporalEntropyUndir={temp_df['spectro_temporal_entropy'].mean(): .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE syllable SET entropyVarUndir={temp_df['entropy_var'].mean(): .4f} WHERE songID= {song_db.id} AND note= '{note}'")
                        elif context == 'D':
                            db.cur.execute(
                                f"UPDATE syllable SET nbNoteDir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                            if len(temp_df) >= nb_note_crit:
                                db.cur.execute(
                                    f"UPDATE syllable SET entropyDir={temp_df['spectral_entropy'].mean() : .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE syllable SET spectroTemporalEntropyDir={temp_df['spectro_temporal_entropy'].mean() : .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE syllable SET entropyVarDir={temp_df['entropy_var'].mean() : .4f} WHERE songID= {song_db.id} AND note= '{note}'")
                    db.conn.commit()

                    # If neither condition meets the number of notes criteria
                    db.cur.execute(
                        f"SELECT nbNoteUndir, nbNoteDir FROM syllable WHERE songID={song_db.id} AND note= '{note}'")
                    nb_notes = [{'U': data[0], 'D': data[1]} for data in db.cur.fetchall()][0]
                    if not (bool(nb_notes['U']) or bool(nb_notes['D'])):
                        db.cur.execute(f"DELETE FROM syllable WHERE songID= {song_db.id} AND note= '{note}'")
                    db.conn.commit()

            # Save df to csv
            if "save_path2" in locals():
                df = df.rename_axis(index='index')
                df.to_csv(save_path2 / ('-'.join(save_path2.stem.split('-')[1:]) + '.csv'), index=True, header=True)

    if update_db:
        db.to_csv('syllable')
        db.to_csv('individual_syllable')

    print('Done!')


if __name__ == '__main__':

    from database.load import create_db, DBInfo, ProjectLoader
    from util import save

    # Parameter
    update = False  # update or make a new cache file for a class object
    save_fig = False
    view_folder = False  # view the folder where figures are stored
    update_db = True  # save results to DB
    fig_ext = '.png'  # .png or .pdf
    nb_note_crit = 10

    # Make save path
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Entropy', add_date=False)

    # Create & Load database
    if update_db:
        # Assumes that song, ff database have been created
        create_db('create_individual_syllable.sql')  # All song syllables
        create_db('create_syllable.sql')  # All song syllables averaged

    # SQL statement
    # query = "SELECT * FROM song WHERE birdID='b70r38'"
    # query = "SELECT * FROM song WHERE id<=2"
    query = "SELECT * FROM song"

    get_entropy(query,
                update=update,
                nb_note_crit=nb_note_crit,
                save_fig=save_fig,
                view_folder=view_folder,
                update_db=update_db,
                fig_ext=fig_ext)

    # Plot values across days
    from deafening.plot import plot_across_days_per_note

    # Load database
    # query = f"""SELECT syl.*, song.taskSession, song.taskSessionDeafening, song.taskSessionPostDeafening, song.dph, song.block10days
    # FROM syllable syl INNER JOIN song ON syl.songID = song.id WHERE syl.nbNoteUndir >= {nb_note_crit}"""
    #
    # df = ProjectLoader().load_db().to_dataframe(query)
    # df.set_index('syllableID')

    # # Spectral Entropy
    # plot_across_days_per_note(df, x='taskSessionDeafening', y='entropyUndir',
    #                           x_label='Days from deafening',
    #                           y_label='Spectral Entropy',
    #                           title=f"Spectral Entropy (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='Spectral_entropy_across_days',
    #                           xlim=[-40, 80], ylim=[0.2, 1],
    #                           vline=0,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )
    #
    # # Spectro-temporal Entropy
    # plot_across_days_per_note(df, x='taskSessionDeafening', y='spectroTemporalEntropyUndir',
    #                           x_label='Days from deafening',
    #                           y_label='Spectro temporal Entropy',
    #                           title=f"Spectrotemporal Entropy (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='Spectro_temporal_entropy_across_days',
    #                           xlim=[-40, 80], ylim=[0.2, 1],
    #                           vline=0,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )
    #
    # # Entropy variance
    # plot_across_days_per_note(df, x='taskSessionDeafening', y='entropyVarUndir',
    #                           x_label='Days from deafening',
    #                           y_label='Entropy variance',
    #                           title=f"Entropy variance (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='EV_across_days',
    #                           xlim=[-40, 80], ylim=[0, 0.04],
    #                           vline=0,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )

    # # Plot normalized values
    # from analysis.functions import add_pre_normalized_col
    #
    # df_norm = add_pre_normalized_col(df, 'entropyUndir', 'entropyUndirNorm')
    # df_norm = add_pre_normalized_col(df_norm, 'entropyDir', 'entropyDirNorm')
    #
    # df_norm = add_pre_normalized_col(df_norm, 'spectroTemporalEntropyUndir', 'spectroTemporalEntropyUndirNorm')
    # df_norm = add_pre_normalized_col(df_norm, 'spectroTemporalEntropyDir', 'spectroTemporalEntropyDirNorm')
    #
    # df_norm = add_pre_normalized_col(df_norm, 'entropyVarUndir', 'entropyVarUndirNorm')
    # df_norm = add_pre_normalized_col(df_norm, 'entropyVarDir', 'entropyVarDirNorm')

    # plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='entropyUndirNorm',
    #                           x_label='Days from deafening',
    #                           y_label='Norm. Spectral Entropy',
    #                           title=f"Norm. Entropy variance (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='Spectral_entropy_norm_across_days',
    #                           xlim=[0, 75], ylim=[0.5, 1.5],
    #                           hline=1,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )
    #
    # plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='spectroTemporalEntropyUndirNorm',
    #                           x_label='Days from deafening',
    #                           y_label='Norm. Spectro temporal Entropy',
    #                           title=f"Norm. Spectro temporal Entropy (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='Spectral_entropy_norm_across_days',
    #                           xlim=[0, 75], ylim=[0.5, 1.5],
    #                           hline=1,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )
    #
    # plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='entropyVarUndirNorm',
    #                           x_label='Days from deafening',
    #                           y_label='Norm. Entropy variance',
    #                           title=f"Norm. Entropy variance (nb of notes >= {nb_note_crit}) Undir",
    #                           fig_name='Entropy_variance_norm_across_days',
    #                           xlim=[0, 75], ylim=[0, 2.5],
    #                           hline=1,
    #                           view_folder=True,
    #                           save_fig=False,
    #                           save_path=save_path
    #                           )

    # Compare conditional means of CV of FF
    # from results.plot import plot_paired_data
    #
    # plot_paired_data(df_norm, x='taskName', y='entropyUndir',
    #                      x_label=None, y_label="Spectral Entropy",
    #                      y_lim=[0, 1.2],
    #                      view_folder=True,
    #                      fig_name='Spectral_entropy_comparison',
    #                      save_fig=False,
    #                      save_path=save_path,
    #                      fig_ext='.png'
    #                      )
    #
    # plot_paired_data(df_norm, x='taskName', y='spectroTemporalEntropyUndir',
    #                      x_label=None, y_label='Spectro temporal Entropy',
    #                      y_lim=[0, 1.2],
    #                      view_folder=True,
    #                      fig_name='Spectro_temporal_entropy_across_days',
    #                      save_fig=False,
    #                      save_path=save_path,
    #                      fig_ext='.png'
    #                      )
    #
    # plot_paired_data(df_norm, x='taskName', y='entropyVarUndir',
    #                      x_label=None, y_label="Entropy variance",
    #                      y_lim=[0, 0.03],
    #                      view_folder=True,
    #                      fig_name='EV_across_days',
    #                      save_fig=False,
    #                      save_path=save_path,
    #                      fig_ext='.png'
    #
    #

    # df_mean.to_csv(save_path / 'df_mean.csv', index=False, header=True)
    #

    # # Compare post-deafening values relative to pre-deafening baseline (1)
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import scipy.stats as stats
    # from util.draw import remove_right_top
    #
    # df_mean = df.groupby(['birdID', 'note', 'taskName']).mean().reset_index()
    # df_mean = df_mean.query('taskName== "Postdeafening"')
    #
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0] = sns.stripplot(ax=axes[0], x=df_mean["taskName"], y=df_mean["entropyUndirNorm"],
    #                         color='k', jitter=0.05)
    # axes[0] = sns.boxplot(ax=axes[0], y=df_mean["entropyUndirNorm"],
    #                       width=0.2, color='w', showfliers = False)
    # axes[0].set_ylim([0, 1.5])
    # axes[0].set_title('Norm. Spectral Entropy')
    # axes[0].set_xlabel('')
    # axes[0].axhline(y=1, color='m', ls='--', lw=0.5)
    # remove_right_top(axes[0])
    #
    # # One-sample t-test (one-tailed)
    # statistics = stats.ttest_1samp(a=df_mean["entropyUndirNorm"].dropna(), popmean=1, alternative='greater')
    # msg = f"t({len(df_mean['entropyUndirNorm'].dropna())-1})=" \
    #       f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
    # axes[0].text(-0.25, 0.1, msg, fontsize=12)
    #
    #
    # axes[1] = sns.stripplot(ax=axes[1], x=df_mean["taskName"], y=df_mean["spectroTemporalEntropyUndirNorm"],
    #                         color='k', jitter=0.05)
    # axes[1] = sns.boxplot(ax=axes[1], y=df_mean["spectroTemporalEntropyUndirNorm"],
    #                       width=0.2, color='w', showfliers = False)
    # axes[1].set_ylim([0, 1.5])
    # axes[1].set_title('Norm. Spectro-temporal Entropy')
    # axes[1].set_xlabel('')
    # axes[1].axhline(y=1, color='m', ls='--', lw=0.5)
    # remove_right_top(axes[1])
    #
    # statistics = stats.ttest_1samp(a=df_mean["spectroTemporalEntropyUndirNorm"].dropna(), popmean=1, alternative='greater')
    # msg = f"t({len(df_mean['spectroTemporalEntropyUndirNorm'].dropna())-1})=" \
    #       f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
    # axes[1].text(-0.25, 0.1, msg, fontsize=12)
    #
    # axes[2] = sns.stripplot(ax=axes[2], x=df_mean["taskName"], y=df_mean["entropyVarUndirNorm"],
    #                         color='k', jitter=0.05)
    # axes[2] = sns.boxplot(ax=axes[2], y=df_mean["entropyVarUndirNorm"],
    #                       width=0.2, color='w', showfliers = False)
    # axes[2].set_ylim([0, 1.5])
    # axes[2].set_title('Norm. Entropy Variance')
    # axes[2].set_xlabel('')
    # axes[2].axhline(y=1, color='m', ls='--', lw=0.5)
    # remove_right_top(axes[2])
    #
    # statistics = stats.ttest_1samp(a=df_mean["entropyVarUndirNorm"].dropna(), popmean=1, alternative='less')
    # msg = f"t({len(df_mean['entropyVarUndirNorm'].dropna())-1})=" \
    #       f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
    # axes[2].text(-0.25, 0.1, msg, fontsize=12)
    #
    # plt.show()

