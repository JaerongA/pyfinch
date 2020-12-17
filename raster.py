# Practice creating an audio object

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

query = "SELECT * FROM cluster WHERE id == 9"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)

rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = 0.5  # text height
font_size = 12

for row in cur.fetchall():

    # ci = ClusterInfo(row)
    # ci.load_events()
    mi = MotifInfo(row, update=True)

    # Plot spectrogram & peri-event histogram (Just the first rendition)
    for onset, offset in zip(mi.onsets, mi.offsets):

        # Convert from string to array of floats
        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        # Motif start and end
        start = onset[0] - peth_parm['buffer']
        end = offset[-1] + peth_parm['buffer']

        # Get spectrogram
        audio = AudioData(row).extract([start, end])
        audio.spectrogram(freq_range=freq_range)

        # Plot figure
        fig = plt.figure(figsize=(6,6.5))
        plt.suptitle(mi.name, y=.95)
        gs = gridspec.GridSpec(9, 1)
        gs.update(hspace=0.2)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3])
        ax_spect.pcolormesh(audio.timebins * 1E3 - peth_parm['buffer'], audio.freqbins, audio.spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        # ax_spect.set_xlabel('Time (ms)', fontsize=font_size)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0], sharex= ax_spect)
        note_dur = offset - onset  # syllable duration
        onset -= onset[0]  # start from 0
        offset = onset + note_dur

        for i, syl in enumerate(mi.motif):
            # Mark syllables
            rectangle = plt.Rectangle((onset[i], rec_yloc), note_dur[i], 0.2,
                                      linewidth=1, alpha = 0.5, edgecolor='k', facecolor=syl_color['Motif'][i])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onset[i] + (offset[i] - onset[i]) / 2), text_yloc, syl, size=font_size)
        ax_syl.axis('off')

        break

    # Plot raster
    line_offsets = np.arange(0.5,len(mi))
    list_zip = zip(mi.spk_ts_warp, mi.onsets, line_offsets)
    ax_raster = plt.subplot(gs[4:-1], sharex= ax_spect)

    for motif_ind, (spk_ts_warp, onset, line_offset) in enumerate(list_zip):
        # print(spk_ts_warp)
        spk = spk_ts_warp - float(onset[0])
        # spk = spk_ts_warp - float(onset[0])
        # print(len(spk))
        print("spk ={}, nb = {}".format(spk, len(spk)))
        print('')

        ax_raster.eventplot(spk, colors='k', lineoffsets=line_offset,
                     linelengths=1, orientation='horizontal')

        k = 1  # index for setting the motif color

        for i, dur in enumerate(mi.median_durations):

            if i == 0:
                rectangle = plt.Rectangle((0, motif_ind), dur, rec_height,
                                          # fill=True,
                                          linewidth=1,
                                          alpha=0.05,
                                          facecolor=syl_color['Motif'][i])
            elif not i%2:
                # print("i is {}, color is {}".format(i, i-k))
                rectangle = plt.Rectangle((sum(mi.median_durations[:i]), motif_ind), mi.median_durations[i], rec_height,
                                          # fill=True,
                                          linewidth=1,
                                          alpha=0.05,
                                          facecolor=syl_color['Motif'][i-k])
                k += 1

            ax_raster.add_patch(rectangle)

    ax_raster.set_ylim(0,len(mi))
    ax_raster.set_ylabel('Trial #', fontsize=font_size)
    ax_raster.set_xlabel('Time (ms)', fontsize=font_size)
    remove_right_top(ax_raster)

    plt.show()