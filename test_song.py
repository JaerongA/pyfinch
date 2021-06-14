"""
By Jaerong
Song analysis
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile

from analysis.parameters import note_buffer, freq_range
from analysis.song import *
from database.load import ProjectLoader, DBInfo
from util import save
from analysis.functions import find_str, read_not_mat
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Parameter
from util.draw import remove_right_top

normalize = False  # normalize correlogram
update = False
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

    si = SongInfo(path, name, update=update)  # song object

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

    ff_info = {data[0] : {'parameter' :data[1],
                          'crit' : data[2],
                          'low' : data[3],  # lower limit of frequency
                          'high' :data[4],  # upper limit of frequency
                          'duration' : data[5]} for data in db.cur.fetchall()  # ff duration
               }

    for file in si.files:

        print(f'Loading... {file.name}')

        # Loop through the notes
        note_ind = 0

        # Load audio object with info from .not.mat files
        ai = AudioInfo(file)
        ai.load_notmat()

        for note, onset, offset in zip(ai.syllables, ai.onsets, ai.offsets):

            if note not in ff_info.keys(): continue  # skip if note has no FF portion
            else:
                note_ind += 1

            # Note start and end
            duration = offset - onset

            # Get spectrogram
            timestamp, data = ai.extract([onset, offset]) # Extract data within the range

            spect_time, spect, spect_freq = ai.spectrogram(timestamp, data)

            # Plot figure
            fig = plt.figure(figsize=(5, 3), dpi=500)
            fig_name = f"{file.name}, note#{note_ind} - {note}"

            plt.suptitle(fig_name, y=.93, fontsize=10)
            gs = gridspec.GridSpec(4, 5)

            # Plot spectrogram
            ax_spect = plt.subplot(gs[1:3, 0:3])
            spect_time = spect_time - spect_time[0] # starts from zero
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
            ax_spect.set_xlabel('Time (ms)', fontsize=10)
            ax_spect.set_ylabel('Frequency (Hz)', fontsize=10)
            plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])

            # Mark FF portion

            # if syl_segment == 1
            # note_length=end_time-start_time; % note_length is in seconds
            # temp_start_time=start_time+(note_length * (percent_from_start / 100));
            # seg_start_time=temp_start_time;
            # seg_end_time=temp_start_time + FF_Duration;
            # elseif syl_segment == 2 % (ms from the start)
            # temp_start_time=start_time+(ms_from_start / 1000); % ( in seconds)
            # seg_start_time=temp_start_time; % sample centered on time entered
            # seg_end_time=(seg_start_time+ FF_Duration); % dur of segment determined by user
            # elseif syl_segment == 3
            # temp_start_time=end_time-(ms_from_end / 1000); % ( in seconds)
            # seg_start_time=temp_start_time; % sample centered on time entered
            # seg_end_time=(seg_start_time+ FF_Duration); % dur of segment determined by user
            # end

            # Get FF onset and offset based on the parameters from DB
            if ff_info[note]['crit'] == 'percent_from_start':
                ff_onset = onset + (duration * (ff_info[note]['parameter'] / 100))
                ff_offset = ff_onset + ff_info[note]['duration']

            _, data = ai.extract([ff_onset, ff_offset])  # Extract data within the range

            # Mark FF
            ax_spect.axvline(x=ff_onset - onset, color='b', linewidth=0.5)
            ax_spect.axvline(x=ff_offset - onset, color='b', linewidth=0.5)

            # Get FF from the FF segment
            np.cov(data)


            break
        break

    plt.show()

    if update_db:
        db.cur.execute("UPDATE song SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbBoutsUndir=?, nbBoutsDir=? WHERE id=?", (nb_bouts['U'], nb_bouts['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbMotifsUndir=?, nbMotifsDir=? WHERE id=?", (nb_motifs['U'], nb_motifs['D'], song_db.id))
        db.cur.execute("UPDATE song SET meanIntroUndir=?, meanIntroDir=? WHERE id=?", (mean_nb_intro_notes['U'], mean_nb_intro_notes['D'], song_db.id))
        db.cur.execute("UPDATE song SET songCallPropUndir=?, songCallPropDir=? WHERE id=?", (song_call_prop['U'], song_call_prop['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationUndir=?, motifDurationDir=? WHERE id=?", (motif_dur['mean']['U'], motif_dur['mean']['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationCVUndir=?, motifDurationCVDir=? WHERE id=?", (motif_dur['cv']['U'], motif_dur['cv']['D'], song_db.id))
    # else:
    #     print(nb_files, nb_bouts, nb_motifs, mean_nb_intro_notes, song_call_prop, motif_dur)

print('Done!')