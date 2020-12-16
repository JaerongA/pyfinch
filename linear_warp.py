# Get raster & peth

from database.load import ProjectLoader
from spike.analysis import *
from spike.parameters import *
from scipy.io import wavfile
from song.parameters import *
from pathlib import Path
from spike.load import read_rhd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from util.spect import *
from util.draw import *
import math



query = "SELECT * FROM cluster WHERE id == 50"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)


rec_yloc = 0.05
rec_height = 0.3  # syllable duration rect
text_yloc = rec_height + 0.2 # text height
font_size = 10






for row in cur.fetchall():

    ci = ClusterInfo(row)
    ci.load_events()
    mi = MotifInfo(row)


    note_dur = np.empty((len(mi), len(mi.motif)))  # nb of motifs x nb of notes
    gap_dur = np.empty((len(mi), len(mi.motif)-1))  # nb of motifs x nb of notes - 1
    # Plot spectrogram & peri-event histogram
    list_zip = zip(mi.onsets, mi.offsets, mi.spk_ts)


    for motif_ind, (onset, offset, spk_ts) in enumerate(list_zip):

        # Convert from string to array of floats
        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        note_dur[motif_ind] = offset - onset

        # Gap duration
        for gap_ind in range(0, len(mi.motif)-1):
            gap_dur[motif_ind, gap_ind] = onset[gap_ind+1] - offset[gap_ind]

    median_note_dur = np.median(note_dur, axis=0)
    median_gap_dur = np.median(gap_dur, axis=0)

    # Piecewise linear warping
    list_zip = zip(note_dur, gap_dur, mi.onsets, mi.offsets, mi.spk_ts)

    for motif_ind, (note_dur, gap_dur, onset, offset, spk_ts) in enumerate(list_zip):  # per motif

        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        for i in range(0, 2 * (len(mi.motif))):
            note_run = math.ceil(i/2)
            if i % 2:
                is_note = True
            else:  # gap
                is_note = False

            if is_note:
                for note_ind, dur in enumerate(note_dur):  # per note
                    ratio = median_note_dur[note_ind] / dur
                    diff = onset[note_ind] - onset[0]
                    origin = sum(median_note_dur[:note_ind+1]) + sum(median_gap_dur[:note_ind+1])
                    ind, spk_ts_new = extract(spk_ts, [onset[note_ind], offset[note_ind]])
                    print(ind)
                    new_ts = ((ratio * ((spk_ts_new - onset[0]) - diff)) + origin) + onset[0]
                    spk_ts[ind] = spk_ts_new
                    # break









# a = np.asarray(mi.contexts)
# ind = np.where(a=='U')
# peth[ind]