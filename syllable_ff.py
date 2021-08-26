"""
By Jaerong
FF analysis
"""


def get_syllable_ff(query,
                    update=False,
                    nb_note_crit=None,
                    save_fig=None,
                    view_folder=False,
                    update_db=False,
                    fig_ext='.png'):

    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from analysis.functions import get_ff
    from analysis.parameters import note_buffer, freq_range
    from analysis.song import AudioInfo, SongInfo
    from util import save
    from util.draw import remove_right_top


    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        si = SongInfo(path, name, update=update)  # song object

        # Fundamental Frequency analysis
        # Retrieve data from ff database
        db.execute(
            f"SELECT ffNote, ffParameter, ffCriterion, ffLow, ffHigh, ffDuration, harmonic FROM ff WHERE birdID='{song_db.birdID}'")

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

        nb_files = si.nb_files
        nb_bouts = si.nb_bouts(song_db.songNote)
        nb_motifs = si.nb_motifs(song_db.motif)

        # mean_nb_intro_notes = si.mean_nb_intro(song_db.introNotes, song_db.songNote)
        # song_call_prop = si.song_call_prop(song_db.introNotes, song_db.songNote)
        # mi = si.get_motif_info(song_db.motif)  # Get motif info
        # motif_dur = mi.get_motif_duration()  # Get mean motif duration &  CV per context

        df = pd.DataFrame()  # Store results here

        note_ind1 = -1  # note index across the session
        for file in si.files:

            # print(f'Loading... {file.name}')
            # Loop through the notes
            note_ind2 = -1  # note index within a file

            # Load audio object with info from .not.mat files
            ai = AudioInfo(file)
            ai.load_notmat()

            for note, onset, offset in zip(ai.syllables, ai.onsets, ai.offsets):
                if note not in set([note[0] for note in ff_info.keys()]): continue  # skip if note has no FF portion

                for i in range(0, [note[0] for note in ff_info.keys()].count(
                        note)):  # if more than one FF can be detected in a single note

                    # Note start and end
                    note_ind1 += 1  # note index across the session
                    note_ind2 += 1  # note index within a file
                    if [note[0] for note in ff_info.keys()].count(note) >= 2:
                        ff_note = f'{note}{i + 1}'
                    else:
                        ff_note = note

                    # if note_ind1 != 1:
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

                    _, data = ai.extract([ff_onset, ff_offset])  # Extract data within the range

                    # Calculate fundamental frequency
                    ff = get_ff(data, ai.sample_rate,
                                ff_info[ff_note]['low'], ff_info[ff_note]['high'],
                                ff_info[ff_note]['harmonic'])

                    if not ff:  # skip the note if the ff is out of the expected range
                        continue

                    # Organize results per song session
                    temp_df = pd.DataFrame({'note': [ff_note], 'context': [ai.context], 'ff': [ff]})
                    df = df.append(temp_df, ignore_index=True)

                    if save_fig:
                        # Parameters
                        font_size = 8
                        # Get spectrogram
                        timestamp, data = ai.extract([onset, offset])  # Extract data within the range
                        spect_time, spect, spect_freq = ai.spectrogram(timestamp, data)

                        # Plot figure
                        fig = plt.figure(figsize=(4, 3), dpi=300)
                        fig_name = f"{note_ind1 :04} - {file.name}, note#{note_ind2} - {ff_note}"

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
                        x_range = np.array([freq_range[0], 1000, 2000, 3000, 4000, 5000, 6000, 7000, freq_range[-1]])
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
                        ax_txt.text(txt_xloc, txt_yloc, f"{ff_info[ff_note]['parameter']} {ff_info[ff_note]['crit']}",
                                    fontsize=font_size)
                        txt_yloc -= txt_offset
                        ax_txt.text(txt_xloc, txt_yloc, f"ff duration = {ff_info[ff_note]['duration']} ms", fontsize=font_size)
                        txt_yloc -= txt_offset
                        ax_txt.text(txt_xloc, txt_yloc, f"ff = {ff} Hz", fontsize=font_size)

                        # Save results
                        save_path2 = save.make_dir(save_path, si.name, add_date=False)
                        save.save_fig(fig, save_path2, fig_name, view_folder=view_folder, fig_ext=fig_ext)

                    # Update ff_results db with note info
                    if update_db:
                        # Fill in song info
                        query = f"INSERT OR IGNORE INTO ff_result (songID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, block10days, note) " \
                                f"VALUES({song_db.id}, '{song_db.birdID}', '{song_db.taskName}', {song_db.taskSession}, {song_db.taskSessionDeafening}, {song_db.taskSessionPostDeafening}, {song_db.block10days}, '{ff_note}')"
                        db.cur.execute(query)
                        db.conn.commit()

        # Save df to csv
        if "save_path2" in locals():
            df = df.rename_axis(index='index')
            df.to_csv(save_path2 / ('-'.join(save_path2.stem.split('-')[1:]) + '.csv'), index=True, header=True)

        # Save results to ff_results db
        if not df.empty:
            if update_db:
                for note in df['note'].unique():
                    for context in df['context'].unique():
                        temp_df = df[(df['note'] == note) & (df['context'] == context)]

                        ff_cv = temp_df['ff'].std() / temp_df['ff'].mean() * 100

                        if len(temp_df) >= nb_note_crit:
                            if context == 'U':
                                db.cur.execute(
                                    f"UPDATE ff_result SET nbNoteUndir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_result SET ffMeanUndir={temp_df['ff'].mean() :1.3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_result SET ffUndirCV={ff_cv : .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                            elif context == 'D':
                                db.cur.execute(
                                    f"UPDATE ff_result SET nbNoteDir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_result SET ffMeanDir={temp_df['ff'].mean() :1.3f} WHERE songID= {song_db.id} AND note= '{note}'")
                                db.cur.execute(
                                    f"UPDATE ff_result SET ffDirCV={ff_cv : .3f} WHERE songID= {song_db.id} AND note= '{note}'")

                    # If neither condition meets the number of notes criteria
                    db.cur.execute(
                        f"SELECT nbNoteUndir, nbNoteDir FROM ff_result WHERE songID={song_db.id} AND note= '{note}'")
                    nb_notes = [{'U': data[0], 'D': data[1]} for data in db.cur.fetchall()][0]
                    if not (bool(nb_notes['U']) or bool(nb_notes['D'])):
                        db.cur.execute(f"DELETE FROM ff_result WHERE songID= {song_db.id} AND note= '{note}'")
                    db.conn.commit()

    if update_db:
        db.to_csv('ff_result')

    print('Done!')


def plot_across_days(df, x, y,
                     x_label=None,
                     y_label=None,
                     title=None, fig_name=None,
                     xlim=None, ylim=None,
                     plot_baseline=False,
                     view_folder=True,
                     save_fig=True,
                     fig_ext='.png'
                     ):

    # Load database
    import seaborn as sns
    import matplotlib.pyplot as plt
    from util.draw import remove_right_top

    # Plot the results
    circ_size = 1

    bird_list = sorted(set(df['birdID'].to_list()))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)

    fig.get_axes()[0].annotate(f"{title}", (0.5, 0.97),
                               xycoords='figure fraction',
                               ha='center',
                               fontsize=16)
    axes = axes.ravel()

    for bird, ax_ind in zip(bird_list, range(len(bird_list))):

        temp_df = df.loc[df['birdID'] == bird]
        sns.lineplot(x=x, y=y, hue='note',
                     data=temp_df, ci=None, marker='o', mew=circ_size, ax=axes[ax_ind])
        remove_right_top(axes[ax_ind])
        axes[ax_ind].set_title(bird)
        if ax_ind >= 5:
            axes[ax_ind].set_xlabel(x_label)
        else:
            axes[ax_ind].set_xlabel('')

        if (ax_ind == 0) or (ax_ind == 5):
            axes[ax_ind].set_ylabel(y_label)
        else:
            axes[ax_ind].set_ylabel('')

        if xlim:
            axes[ax_ind].set_xlim(xlim)
        if ylim:
            axes[ax_ind].set_ylim(ylim)

        if plot_baseline:
            axes[ax_ind].axhline(y=1, color='k', ls='--', lw=0.5)

    if save_fig:
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()


def normalize_from_pre(df, var_name: str, note: str):
    """Normalize post-deafening values using pre-deafening values"""
    pre_val = df.loc[(df['note'] == note) & (df['taskName'] == 'Predeafening')][var_name]
    pre_val = pre_val.mean()

    post_val = df.loc[(df['note'] == note) & (df['taskName'] == 'Postdeafening')][var_name]
    norm_val = post_val / pre_val

    return norm_val


def add_pre_normalized_col(df, col_name_to_normalize, col_name_to_add, csv_name=None, save_csv=False):
    """Normalize relative to pre-deafening mean"""
    import numpy as np

    df[col_name_to_add] = np.nan

    bird_list = sorted(set(df['birdID'].to_list()))
    for bird in bird_list:

        temp_df = df.loc[df['birdID'] == bird]
        note_list = temp_df['note'].unique()

        for note in note_list:
            norm_val = normalize_from_pre(temp_df, col_name_to_normalize, note)
            add_ind = temp_df.loc[(temp_df['note'] == note) & (temp_df['taskName'] == 'Postdeafening')].index
            df.loc[add_ind, col_name_to_add] = norm_val

    if save_csv:
        df.to_csv(save_path / csv_name, index=False, header=True)

    return df

if __name__ == '__main__':

    from database.load import create_db, DBInfo, ProjectLoader
    from util import save

    # Parameter
    save_fig = True  # save spectrograms with FF
    view_folder = False  # view the folder where figures are stored
    update_db = True  # save results to DB
    fig_ext = '.png'  # .png or .pdf
    nb_note_crit = 10

    # Make save path
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'FF', add_date=False)

    # Create & Load database
    if update_db:
        # Assumes that song, ff database have been created
        db = create_db('create_ff_result.sql')

    # SQL statement
    # query = "SELECT * FROM song WHERE id>=80"
    #
    # get_syllable_ff(query,
    #                 nb_note_crit=nb_note_crit,
    #                 save_fig=save_fig,
    #                 view_folder=view_folder,
    #                 update_db=update_db,
    #                 fig_ext=fig_ext)

    # Load database
    df = ProjectLoader().load_db().to_dataframe(f"SELECT * FROM ff_result")
    df_norm = add_pre_normalized_col(df, 'ffUndirCV', 'ffUndirCVNorm')
    df_norm = add_pre_normalized_col(df_norm, 'ffDirCV', 'ffDirCVNorm', csv_name='ff_results.csv', save_csv=True)

    # Plot FF per day
    # Parameters
    fr_criteria = 10

    # plot_across_days(df_norm, x='taskSessionDeafening', y='ffUndirCV',
    #                  x_label='Days from deafening',
    #                  y_label='FF',
    #                  title='CV of FF (Undir)', fig_name='FF_across_days',
    #                  xlim=[-20, 40], ylim=[0, 5],
    #                  view_folder=True,
    #                  save_fig=False,
    #                  )

    plot_across_days(df_norm, x='taskSessionDeafening', y='ffUndirCVNorm',
                     x_label='Days from deafening',
                     y_label='Norm. FF',
                     title='CV of FF (Undir)', fig_name='FF_across_days',
                     xlim=[0, 31],
                     ylim=[0, 3],
                     plot_baseline=True,
                     view_folder=True,
                     save_fig=False,
                     )
