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

for row in cur.fetchall():

    ci = ClusterInfo(row)
    ci.load_events()
    mi = MotifInfo(row)
    # bi = BaselineInfo(row, update=True)

    # Plot raw results

    start = float(mi.onsets[0][0]) - peth['buffer']
    end = float(mi.offsets[0][-1]) + peth['buffer']

    audio = AudioData(row).extract([start, end])
    audio.spectrogram(freq_range=freq_range)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.pcolormesh(audio.timebins, audio.freqbins, audio.spect,
                        cmap='hot_r',
                        norm=colors.SymLogNorm(linthresh=0.05,
                                               linscale=0.03,
                                               vmin=0.5,
                                               vmax=100
                                               ))

    remove_right_top(ax)
    ax.set_ylim(300, 8000)

    plt.show()
