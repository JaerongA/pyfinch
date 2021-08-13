"""
By Jaerong
FF analysis
"""

from analysis.parameters import note_buffer, freq_range, nb_note_crit
from analysis.song import AudioInfo, SongInfo
from database.load import ProjectLoader, DBInfo
from util import save
from analysis.functions import para_interp, get_ff
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as smt
from scipy.signal import find_peaks
from util.draw import remove_right_top
import pandas as pd

# Parameter
update = False  # update or make a new cache file
save_fig = True  # save spectrograms with FF
view_folder = False  # view the folder where figures are stored
update_db = True  # save results to DB
fig_ext = '.png'  # .png or .pdf
txt_offset = 0.2
font_size = 8

# Load database
db = ProjectLoader().load_db()

# Make database
with open('database/create_song_table.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# Parameter values should have been filled already
with open('database/create_ff.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# Results will be stored here
with open('database/create_ff_result.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# Make save path
save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'FF', add_date=False)

# SQL statement
# query = "SELECT * FROM song WHERE birdID='b70r38'"
# query = "SELECT * FROM song WHERE id=2"
query = "SELECT * FROM song WHERE id >= 17"
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
        f"SELECT ffNote, ffParameter, ffCriterion, ffLow, ffHigh, ffDuration FROM ff WHERE birdID='{song_db.birdID}'")

    ff_info = {data[0]: {'parameter': data[1],
                         'crit': data[2],
                         'low': data[3],  # lower limit of frequency
                         'high': data[4],  # upper limit of frequency
                         'duration': data[5]} for data in db.cur.fetchall()  # ff duration
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

        print(f'Loading... {file.name}')
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
                #     continue

                duration = offset - onset

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

                # Mark FF
                ax_spect.axvline(x=ff_onset - onset, color='b', linewidth=0.5)
                ax_spect.axvline(x=ff_offset - onset, color='b', linewidth=0.5)

                # Calculate fundamental frequency
                ff = get_ff(data, ai.sample_rate,
                            ff_info[ff_note]['low'], ff_info[ff_note]['high'],
                            ff_harmonic=ff_info[ff_note]['harmonic'])

                if not ff:  # skip the note if the ff is out of the expected range
                    continue

                # Mark estimated FF
                ax_spect.axhline(y=ff, color='g', ls='--', lw=0.8)

                # Print out text results
                txt_xloc = -1.2
                txt_yloc = 0.8

                ax_txt = plt.subplot(gs[1:, -1])
                ax_txt.set_axis_off()  # remove all axes
                ax_txt.text(txt_xloc, txt_yloc, f"{ff_info[ff_note]['parameter']} {ff_info[ff_note]['crit']}",
                            fontsize=font_size)
                txt_yloc -= txt_offset
                ax_txt.text(txt_xloc, txt_yloc, f"ff duration = {ff_info[ff_note]['duration']} ms", fontsize=font_size)
                txt_yloc -= txt_offset
                ax_txt.text(txt_xloc, txt_yloc, f"ff = {ff} Hz", fontsize=font_size)

                # Save results
                if save_fig:
                    save_path2 = save.make_dir(save_path, si.name, add_date=False)
                    save.save_fig(fig, save_path2, fig_name, view_folder=view_folder, fig_ext=fig_ext)

                # Organize results per song session
                temp_df = pd.DataFrame({'note': [ff_note], 'context': [ai.context], 'ff': [ff]})
                df = df.append(temp_df, ignore_index=True)

                # Update ff_results db with note info
                if update_db:
                    # Fill in song info
                    query = f"INSERT OR IGNORE INTO ff_result (songID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, block10days, note) " \
                            f"VALUES({song_db.id}, '{song_db.birdID}', '{song_db.taskName}', {song_db.taskSession}, {song_db.taskSessionDeafening}, {song_db.taskSessionPostDeafening}, {song_db.block10days}, '{ff_note}')"
                    db.cur.execute(query)
                    db.conn.commit()

    # Save results to ff_results db
    if not df.empty:
        if update_db:
            for note in df['note'].unique():
                for context in df['context'].unique():
                    temp_df = df[(df['note'] == note) & (df['context'] == context)]
                    if len(temp_df) >= nb_note_crit:
                        if context == 'U':
                            db.cur.execute(
                                f"UPDATE ff_result SET nbNoteUndir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                            db.cur.execute(
                                f"UPDATE ff_result SET ffMeanUndir={temp_df['ff'].mean() :1.3f} WHERE songID= {song_db.id} AND note= '{note}'")
                            db.cur.execute(
                                f"UPDATE ff_result SET ffUndirCV={temp_df['ff'].std() / temp_df['ff'].mean() * 100 : .3f} WHERE songID= {song_db.id} AND note= '{note}'")
                        elif context == 'D':
                            db.cur.execute(
                                f"UPDATE ff_result SET nbNoteDir={len(temp_df)} WHERE songID= {song_db.id} AND note= '{note}'")
                            db.cur.execute(
                                f"UPDATE ff_result SET ffMeanDir={temp_df['ff'].mean() :1.3f} WHERE songID= {song_db.id} AND note= '{note}'")
                            db.cur.execute(
                                f"UPDATE ff_result SET ffDirCV={temp_df['ff'].std() / temp_df['ff'].mean() * 100 : .3f} WHERE songID= {song_db.id} AND note= '{note}'")

                # If neither condition meets the number of notes criteria
                db.cur.execute(
                    f"SELECT nbNoteUndir, nbNoteDir FROM ff_result WHERE songID={song_db.id} AND note= '{note}'")
                nb_notes = [{'U': data[0], 'D': data[1]} for data in db.cur.fetchall()][0]
                if not (bool(nb_notes['U']) or bool(nb_notes['D'])):
                    db.cur.execute(f"DELETE FROM ff_result WHERE songID= {song_db.id} AND note= '{note}'")
                db.conn.commit()

        # Save df to csv
        if "save_path2" in locals():
            df = df.rename_axis(index='index')
            df.to_csv(save_path2 / ('-'.join(save_path2.stem.split('-')[1:]) + '.csv'), index=True, header=True)

print('Done!')
