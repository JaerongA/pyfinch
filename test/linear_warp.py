# Get raster & peth

from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
from scipy.io import wavfile
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from util.spect import *
from util.draw import *
import math

query = "SELECT * FROM cluster WHERE id == 9"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)

rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = rec_height + 0.2  # text height
font_size = 10

for row in cur.fetchall():
    ci = ClusterInfo(row)
    ci._load_events()
    ci._load_spk()
    mi = MotifInfo(row, update=True)

    # duration = np.empty((len(mi), len(mi.motif)*2-1))
    #
    # # Plot spectrogram & peri-event histogram
    # list_zip = zip(mi.onsets, mi.offsets, mi.spk_ts)
    #
    #
    # for motif_ind, (onset, offset, spk_ts) in enumerate(list_zip):
    #
    #     # Convert from string to array of floats
    #     onset = np.asarray(list(map(float, onset)))
    #     offset = np.asarray(list(map(float, offset)))
    #
    #     # Calculate note & interval duration
    #     timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
    #     timestamp = sum(timestamp,[])
    #
    #     for i in range(len(timestamp)-1):
    #         duration[motif_ind,i] = timestamp[i+1] - timestamp[i]
    #
    # median_dur = np.median(duration, axis=0)
    #
    # # Piecewise linear warping
    # spk_ts_warped_list = []
    # spk_ts_warped = np.array([], dtype=np.float32)
    #
    # list_zip = zip(duration, mi.onsets, mi.offsets, mi.spk_ts)
    #
    # for motif_ind, (duration, onset, offset, spk_ts) in enumerate(list_zip):  # per motif
    #
    #     onset = np.asarray(list(map(float, onset)))
    #     offset = np.asarray(list(map(float, offset)))
    #
    #     # Calculate note & interval duration
    #     timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
    #     timestamp = sum(timestamp,[])
    #
    #     for i in range(0, len(median_dur)):
    #
    #         ratio = median_dur[i] / duration[i]
    #         diff = timestamp[i] - timestamp[0]
    #
    #         if i == 0:
    #             origin = 0
    #         else:
    #             origin = sum(median_dur[:i])
    #         ind, spk_ts_new = extract(spk_ts, [timestamp[i], timestamp[i+1]])
    #         ts = ((ratio * ((spk_ts_new - timestamp[0]) - diff)) + origin) + timestamp[0]
    #         spk_ts_warped = np.append(spk_ts_warped, ts)
    #
    #     spk_ts_warped_list.append(spk_ts_warped)

# a = np.asarray(mi.contexts)
# ind = np.where(a=='U')
# peth[ind]

    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(111)
    line_offsets = np.arange(0.5,len(mi))

    list_zip = zip(mi.spk_ts, mi.onsets, line_offsets)

    for motif_ind, (spk_ts_warp, onset, line_offset) in enumerate(list_zip):
        # print(spk_ts_warp)
        spk = spk_ts_warp - float(onset[0])
        # spk = spk_ts_warp - float(onset[0])
        # print(len(spk))
        print("spk ={}, nb = {}".format(spk, len(spk)))
        print('')

        ax.eventplot(spk, colors='k', lineoffsets=line_offset,
                     linelengths=1, orientation='horizontal')

        k = 1  # index for setting the motif color

        for i, dur in enumerate(mi.median_durations):

            if i == 0:
                rectangle = plt.Rectangle((0, motif_ind), dur, rec_height,
                                          # fill=True,
                                          linewidth=1,
                                          alpha=0.1,
                                          facecolor=note_color['Motif'][i])
            elif not i%2:
                print("i is {}, color is {}".format(i, i-k))
                rectangle = plt.Rectangle((sum(mi.median_durations[:i]), motif_ind), mi.median_durations[i], rec_height,
                                          # fill=True,
                                          linewidth=1,
                                          alpha=0.1,
                                          facecolor=note_color['Motif'][i - k])
                k += 1

            ax.add_patch(rectangle)

    ax.set_ylim(0,len(mi))
    remove_right_top(ax)
    plt.show()
