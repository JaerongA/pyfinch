"""
By Jaerong
Song analysis
"""

import matplotlib.pyplot as plt

from analysis.parameters import note_buffer, freq_range
from analysis.song import *
from database.load import ProjectLoader, DBInfo
from util import save
from analysis.functions import find_str
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Parameter
from util.draw import remove_right_top

normalize = False  # normalize correlogram
update = True
save_fig = False
update_db = False  # save results to DB
fig_ext = '.png'  # .png or .pdf

# Load database
db = ProjectLoader().load_db()
with open('database/create_song_table.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

with open('database/create_ff.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# SQL statement
query = "SELECT * FROM song WHERE id=1"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():
    # Load song info from db
    song_db = DBInfo(row)
    name, path = song_db.load_song_db()

    si = SongInfo(path, name, update=update)  # cluster object
    audio_data = AudioData(path, update=update)

    nb_files = si.nb_files
    nb_bouts = si.nb_bouts(song_db.songNote)
    nb_motifs = si.nb_motifs(song_db.motif)

    # mean_nb_intro_notes = si.mean_nb_intro(song_db.introNotes, song_db.songNote)
    # song_call_prop = si.song_call_prop(song_db.introNotes, song_db.songNote)
    # mi = si.get_motif_info(song_db.motif)  # Get motif info
    # motif_dur = mi.get_motif_duration()  # Get mean motif duration &  CV per context

    # Fundamental Frequency analysis
    # Retrieve data from ff database
    db.execute(f"SELECT ffNote, ffParameter, ffCriterion, ffLow, ffHigh, ffDuration FROM ff WHERE birdID='{song_db.birdID}'")

    ff_data = {data[0] : {'ff_parameter' :data[1], 'ff_crit' : data[2],
                     'ff_low' : data[3],
                     'ff_high' :data[4],
                     'ff_dur' : data[5]} for data in db.cur.fetchall()}

    for data in ff_data:

        syllable_list = [syllable for syllable, context in zip(si.syllables, si.contexts) if context == 'D']
        onset_list = [onset for onset, context in zip(si.onsets, si.contexts) if context == 'U']

        syllables = ''.join(syllable_list)
        onsets = np.hstack(si.onsets)
        offsets = np.hstack(si.offsets)
        contexts = np.hstack(si.contexts)

        note_ind = np.array(find_str(syllables, ff_note))

        # Find onsets and offsets of the target note
        onsets = onsets[note_ind].astype(np.float)
        offsets = offsets[note_ind].astype(np.float)

        # Loop through the notes
        for note_ind, (note, onset, offset) in enumerate(zip(syllables, onsets, offsets)):

            # Note start and end
            start = onset - note_buffer
            end = offset + note_buffer
            duration = offset - onset

            # Get spectrogram
            audio_data = audio_data.extract([start, end])  # audio object
            audio_data.spectrogram()

            # Plot figure
            fig = plt.figure(figsize=(4, 3), dpi=500)
            fig_name = f"{si.name}, note#{note_ind} - {note}"

            plt.suptitle(fig_name, y=.93, fontsize=10)
            gs = gridspec.GridSpec(4, 4)
            # gs.update(wspace=0.025, hspace=0.05)

            # Plot spectrogram
            ax_spect = plt.subplot(gs[1:3, 0:3])
            audio_data.spect_time = audio_data.spect_time - audio_data.spect_time[0] - note_buffer  # starts from zero
            ax_spect.pcolormesh(audio_data.spect_time, audio_data.spect_freq, audio_data.spect,  # data
                                cmap='hot_r',
                                norm=colors.SymLogNorm(linthresh=0.05,
                                                       linscale=0.03,
                                                       vmin=0.5,
                                                       vmax=100
                                                       ))

            remove_right_top(ax_spect)
            ax_spect.set_xlim(-note_buffer, duration + note_buffer)
            ax_spect.set_ylim(freq_range[0], freq_range[1])
            ax_spect.set_ylabel('Frequency (Hz)', fontsize=10)
            plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
            plt.setp(ax_spect.get_xticklabels(), visible=False)
            plt.show()

            break




    if update_db:
        db.cur.execute("UPDATE song SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbBoutsUndir=?, nbBoutsDir=? WHERE id=?", (nb_bouts['U'], nb_bouts['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbMotifsUndir=?, nbMotifsDir=? WHERE id=?", (nb_motifs['U'], nb_motifs['D'], song_db.id))
        db.cur.execute("UPDATE song SET meanIntroUndir=?, meanIntroDir=? WHERE id=?", (mean_nb_intro_notes['U'], mean_nb_intro_notes['D'], song_db.id))
        db.cur.execute("UPDATE song SET songCallPropUndir=?, songCallPropDir=? WHERE id=?", (song_call_prop['U'], song_call_prop['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationUndir=?, motifDurationDir=? WHERE id=?", (motif_dur['mean']['U'], motif_dur['mean']['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationCVUndir=?, motifDurationCVDir=? WHERE id=?", (motif_dur['cv']['U'], motif_dur['cv']['D'], song_db.id))
    else:
        print(nb_files, nb_bouts, nb_motifs, mean_nb_intro_notes, song_call_prop, motif_dur)

print('Done!')