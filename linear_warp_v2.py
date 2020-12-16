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

    #
    # note_dur = np.empty((len(mi), len(mi.motif)))  # nb of motifs x nb of notes
    # gap_dur = np.empty((len(mi), len(mi.motif)-1))  # nb of motifs x nb of notes - 1

    duration = np.empty((len(mi), len(mi.motif)*2-1))

    # Plot spectrogram & peri-event histogram
    list_zip = zip(mi.onsets, mi.offsets, mi.spk_ts)


    for motif_ind, (onset, offset, spk_ts) in enumerate(list_zip):

        # Convert from string to array of floats
        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        # Calculate note & interval duration
        timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
        timestamp = sum(timestamp,[])

        for i in range(len(timestamp)-1):
            duration[motif_ind,i] = timestamp[i+1] - timestamp[i]

    median_dur = np.median(duration, axis=0)

    # Piecewise linear warping
    spk_ts_warped_list = []
    spk_ts_warped = np.array([], dtype=np.float32)

    list_zip = zip(duration, mi.onsets, mi.offsets, mi.spk_ts)

    for motif_ind, (duration, onset, offset, spk_ts) in enumerate(list_zip):  # per motif

        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        # Calculate note & interval duration
        timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
        timestamp = sum(timestamp,[])

        for i in range(0, len(median_dur)):

            ratio = median_dur[i] / duration[i]
            diff = timestamp[i] - timestamp[0]

            if i == 0:
                origin = 0
            else:
                origin = sum(median_dur[:i])
            ind, spk_ts_new = extract(spk_ts, [timestamp[i], timestamp[i+1]])
            ts = ((ratio * ((spk_ts_new - timestamp[0]) - diff)) + origin) + timestamp[0]
            spk_ts_warped = np.append(spk_ts_warped, ts)

        spk_ts_warped_list.append(spk_ts_warped)









# a = np.asarray(mi.contexts)
# ind = np.where(a=='U')
# peth[ind]