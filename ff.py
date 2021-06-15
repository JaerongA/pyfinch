"""
By Jaerong
FF analysis
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile

from analysis.parameters import note_buffer, freq_range
from analysis.song import *
from database.load import ProjectLoader, DBInfo
from util import save
from analysis.functions import find_str, read_not_mat, para_interp
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as smt
from scipy.signal import find_peaks
from util.draw import remove_right_top
import pandas as pd

# Parameter
normalize = False  # normalize correlogram
update = False
save_fig = True
update_db = False  # save results to DB
fig_ext = '.png'  # .png or .pdf
txt_xloc = -1.2
txt_yloc = 0.8
txt_offset = 0.2
font_size = 8

# Load database
db = ProjectLoader().load_db()
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

    df = pd.DataFrame() # Store results here

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
            fig = plt.figure(figsize=(4, 3), dpi=300)
            fig_name = f"{file.name}, note#{note_ind} - {note}"

            plt.suptitle(fig_name, y=.93, fontsize=font_size)
            gs = gridspec.GridSpec(4, 6)

            # Plot spectrogram
            ax_spect = plt.subplot(gs[1:3, 0:4])
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
            ax_spect.set_xlabel('Time (ms)', fontsize=font_size)
            ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
            x_range = np.array([freq_range[0], 1000, 2000, 3000, 4000, 5000, 6000, 7000, freq_range[-1]])
            plt.yticks(x_range, list(map(str, x_range)), fontsize=5)
            plt.xticks(fontsize=5)

            # Get FF onset and offset based on the parameters from DB
            if ff_info[note]['crit'] == 'percent_from_start':
                ff_onset = onset + (duration * (ff_info[note]['parameter'] / 100))
                ff_offset = ff_onset + ff_info[note]['duration']
            elif ff_info[note]['crit'] == 'ms_from_start':
                ff_onset = onset + (ff_info[note]['parameter'])
                ff_offset = ff_onset + ff_info[note]['duration']
            elif ff_info[note]['crit'] == 'ms_from_end':
                ff_offset = offset - ff_info[note]['parameter']
                ff_onset = ff_offset - ff_info[note]['duration']

            _, data = ai.extract([ff_onset, ff_offset])  # Extract data within the range

            # Mark FF
            ax_spect.axvline(x=ff_onset - onset, color='b', linewidth=0.5)
            ax_spect.axvline(x=ff_offset - onset, color='b', linewidth=0.5)

            # Get FF from the FF segment
            corr = smt.ccf(data, data, adjusted=False)
            corr_win = corr[3: round(ai.sample_rate / ff_info[note]['low'])]

            peak_ind, property = find_peaks(corr_win, height=0)

            # Plot auto-correlation (for debugging)
            # plt.plot(corr_win)
            # plt.plot(peak_ind, corr_win[peak_ind], "x")
            # plt.show()

            for ind in property['peak_heights'].argsort()[::-1][0:1]:  # first two peaks
                if peak_ind[ind] and (peak_ind[ind] < len(corr_win)):  # if the peak is not in first and last indices
                    target_peak_ind = peak_ind[ind]
                    target_peak_amp = corr_win[target_peak_ind - 1: target_peak_ind + 2]  # find the peak using two neighboring values using parabolic interpolation
                    target_peak_ind = np.arange(target_peak_ind-1, target_peak_ind+2)
                    peak, _ = para_interp(target_peak_ind, target_peak_amp)

                period = peak + 2
                ff = round(ai.sample_rate / period, 3)

                if (ff > ff_info[note]['low']) and (ff < ff_info[note]['high']):
                    break
            # Mark estimated FF
            ax_spect.axhline(y=ff, color='g', ls='--', lw=1)

            # Print out text results

            ax_txt = plt.subplot(gs[1:, -1])
            ax_txt.set_axis_off()  # remove all axes
            ax_txt.text(txt_xloc, txt_yloc, f"{ff_info[note]['parameter']} {ff_info[note]['crit']}", fontsize=font_size)
            txt_yloc -= txt_offset
            ax_txt.text(txt_xloc, txt_yloc, f"ff duration = {ff_info[note]['duration']} ms", fontsize=font_size)
            txt_yloc -= txt_offset
            ax_txt.text(txt_xloc, txt_yloc, f"ff = {ff} Hz", fontsize=font_size)
            fig.tight_layout()

            # Save results
            if save_fig:
                save_path = save.make_dir(save_path, si.name, add_date=False)
                save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
            else:
                plt.show()

            # Organize results per song session
            # ff_arr[note][ai.context] = np.append(ff_arr[note][ai.context], ff)
            temp_df = pd.DataFrame({'note': [note], 'context': [ai.context], 'ff': [ff]})
            df = df.append(temp_df, ignore_index=True)

            break
        break

    if update_db:
        query = "INSERT INTO ff_result(songID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, block10days) " \
                "VALUES({}, {}, {}, {}, {}, {}, {})".format(song_db.id, song_db.birdID, song_db.taskName, song_db.taskSession, song_db.taskSessionDeafening, song_db.taskSessionPostDeafening, song_db.block10days)
        db.cur.execute(query)

        db.cur.execute("UPDATE ff_result SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
        db.cur.execute("UPDATE ff_result SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
        db.cur.execute("UPDATE ff_result SET nbBoutsUndir=?, nbBoutsDir=? WHERE id=?", (nb_bouts['U'], nb_bouts['D'], song_db.id))
        db.cur.execute("UPDATE ff_result SET nbMotifsUndir=?, nbMotifsDir=? WHERE id=?", (nb_motifs['U'], nb_motifs['D'], song_db.id))
        # db.cur.execute("UPDATE ff_result SET meanIntroUndir=?, meanIntroDir=? WHERE id=?", (mean_nb_intro_notes['U'], mean_nb_intro_notes['D'], song_db.id))
        # db.cur.execute("UPDATE ff_result SET songCallPropUndir=?, songCallPropDir=? WHERE id=?", (song_call_prop['U'], song_call_prop['D'], song_db.id))
        # db.cur.execute("UPDATE ff_result SET motifDurationUndir=?, motifDurationDir=? WHERE id=?", (motif_dur['mean']['U'], motif_dur['mean']['D'], song_db.id))
        # db.cur.execute("UPDATE ff_result SET motifDurationCVUndir=?, motifDurationCVDir=? WHERE id=?", (motif_dur['cv']['U'], motif_dur['cv']['D'], song_db.id))
    # else:
    #     print(nb_files, nb_bouts, nb_motifs, mean_nb_intro_notes, song_call_prop, motif_dur)

print('Done!')