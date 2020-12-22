"""
By Jaerong
Extracts the audio signal (board_adc_data) from .rhd and convert it into .wav
"""

from load_intan_rhd_format.load_intan_rhd_format import read_rhd
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
from scipy.signal import hilbert
from matplotlib import collections as mc
from song.parameters import *
from util.functions import find_data_path

# def find_data_path():
#     """Find the data dir and return it"""
#     from pathlib import Path
#     from tkinter import Tk
#     from tkinter import filedialog
#     root = Tk()
#     root.withdraw()
#     data_dir = filedialog.askdirectory()
#     return Path(data_dir)


# Specify dir here or search for the dir manually
data_dir = Path(r'H:\Box\Data\Deafening Project\y44r34\Predeafening\20200703\Dir')
try:
    data_dir
except NotADirectoryError:
    data_dir = find_data_path()

rhd_files = [str(rhd) for rhd in data_dir.rglob('*.rhd')]

for rhd in rhd_files:

    file_name = Path(rhd).stem
    # fig_name = Path(file_name).with_suffix('.png')
    fig_name = Path(rhd).with_suffix('.png')
    intan = read_rhd(rhd)  # load the .rhd file
    # print(intan.keys())

    intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0
    nb_ch = intan['amplifier_data'].shape[0]

    fig = plt.figure(figsize=(10, 4), dpi=800)
    fig, ax = plt.subplots(nrows=nb_ch + 1, ncols=1, sharex=True)

    # Plot spectrogram for song
    ax[0].set_title(file_name)
    # ax[0].plot(intan['t_amplifier'], intan['board_adc_data'][0], 'k', linewidth=0.5)  # plot the raw signal
    ax[0].specgram(intan['board_adc_data'][0], Fs=sample_rate['intan'], cmap='jet', scale_by_freq=None)
    ax[0].spines['right'].set_visible(False), ax[0].spines['top'].set_visible(False)
    ax[0].set_ylim(freq_range)
    # ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].text(ax[0].get_xlim()[0]*-0.5, freq_range[0]*0.1, str(freq_range[0]), size=10)
    ax[0].text(ax[0].get_xlim()[0]*-0.5, freq_range[1]*0.95, str(freq_range[1]), size=10)
    ax[0].set_ylabel('Frequency')

    # Set the range of the y-axis
    y_range = [abs(intan['amplifier_data'].min()), abs(intan['amplifier_data'].max())]
    y_range = ceil(max(y_range) / 1E2) * 1E2

    for i, ch in enumerate(intan['amplifier_data']):
        ax[i + 1].plot(intan['t_amplifier'], ch, 'k', linewidth=0.5, clip_on=False)
        # ax[i + 1].set_title(intan['amplifier_channels'][i]['native_channel_name'])
        ax[i + 1].spines['right'].set_visible(False)
        ax[i + 1].spines['left'].set_visible(False)
        ax[i + 1].spines['top'].set_visible(False)
        ax[i + 1].spines['bottom'].set_visible(False)
        ax[i + 1].set_ylabel(intan['amplifier_channels'][i]['native_channel_name'])
        ax[i + 1].set_ylim([-y_range, y_range])
        ax[i + 1].set_xticks([])
        ax[i + 1].set_yticks([])

        if i is nb_ch - 1:  # the bottom plot
            # ax[i + 1].set_xlabel('Time (s)')
            xlim_min = ax[i + 1].get_xlim()[0]
            ylim_min = ax[i + 1].get_ylim()[0] * 0.9
            # ylim_min += ylim_min * 0.9
            line = [[(0, ylim_min), (1, ylim_min)],
                    [(xlim_min, -250), (xlim_min, 250)]]
            lc = mc.LineCollection(line, linewidths=4)
            ax[i + 1].add_collection(lc)

            text_xloc = xlim_min + (xlim_min * 0.55)
            text_yloc = ylim_min + (ylim_min * 0.35)

            ax[i + 1].text(0, text_yloc, '1000 ms', size=6, weight='bold')  # x-axis
            ax[i + 1].text(text_xloc, -250, '500 µV', size=6, weight='bold', rotation=90)  # y-axis

    plt.margins(0.05)
    plt.tight_layout()

    # Save save_fig
    # fig_name = Path(rhd).with_suffix('.png')
    # plt.savefig(fig_name)
    # fig_name = Path(rhd).with_suffix('.pdf')  # vector format
    # plt.savefig(fig_name)
    plt.show()
    # plt.clf()
    break
