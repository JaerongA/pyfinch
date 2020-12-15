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
from util import save
from util.spect import *
from util.draw import *

query = "SELECT * FROM cluster WHERE id == 50"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)



rec_height = 0.1  # syllable duration rect
text_yloc = rec_height + 0.1  # text height
nb_row = 2
nb_cols = 1

for row in cur.fetchall():

    ci = ClusterInfo(row)
    ci.load_events()
    mi = MotifInfo(row)

    # Plot spectrogram & peri-event histogram

    # for onset, offset in zip(mi.onsets, mi.offsets):
    for onset, offset in zip(mi.onsets, mi.offsets):

        # Convert from string to array of floats
        onset = np.asarray(list(map(float, onset)))
        offset = np.asarray(list(map(float, offset)))

        # Motif start and end
        start = onset[0] - peth['buffer']
        end = offset[-1] + peth['buffer']

        # Get spectrogram
        audio = AudioData(row).extract([start, end])
        audio.spectrogram(freq_range=freq_range)

        fig, (ax_syl, ax_spect) = plt.subplots(nb_row, nb_cols, figsize=(7, 3), sharex=True)

        # Plot syllable duration
        syl_dur = offset - onset  # syllable duration
        onset -= onset[0] # start from 0
        offset = onset + syl_dur

        for i, syl in enumerate(mi.motif):

            # Mark syllables
            rectangle = plt.Rectangle((onset[i], 0.01), syl_dur[i], rec_height,
                                      linewidth=1, edgecolor='k', facecolor=syl_color['Motif'][i])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onset[i] + (offset[i] - onset[i]) / 2), text_yloc, syl)

        ax_syl.axis('off')
        ax_syl.text((onset[i] + (offset[i] - onset[i]) / 2), text_yloc, syl)
        # plt.title(mi.name)

        # Plot spectrogram
        ax_spect.pcolormesh(audio.timebins*1E3 - peth['buffer'], audio.freqbins, audio.spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))


        remove_right_top(ax_spect)
        ax_spect.set_ylim(300, 8000)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.margins(x=0, y=0.001)
        # fig.tight_layout()
        fig.suptitle(mi.name)
        plt.show()
        break