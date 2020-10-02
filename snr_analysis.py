"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load, save
from load_intan_rhd_format.load_intan_rhd_format import read_data
import numpy as np
import scipy.io
from spike.parameters import sample_rate
from spike.load import read_spk_txt
import matplotlib.pyplot as plt

query = "SELECT * FROM cluster WHERE id = '22'"
cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():
    cell_name, cell_path = load.cell_info(cell_info)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).mat'))[0]
    channel_info = scipy.io.loadmat(mat_file)
    spk_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).txt'))[0]
    unit_nb = int(cell_info['unit'][-2:])

    # Extract the raw neural trace (from the .mat file)
    raw_trace = channel_info['amplifier_data'][0]

    # Read from the cluster .txt file
    spk_ts, spk_waveform, nb_spk = read_spk_txt(spk_file, unit_nb)

    # Plot the individual waveforms
    plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    x_time = np.arange(0, spk_waveform.shape[1]) / sample_rate * 1E3  # x-axis in miliseconds
    for wave in spk_waveform:
        # print(wave.shape)
        ax.plot(x_time, wave, color='k', lw=0.2)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.plot(x_time, np.nanmean(spk_waveform, axis=0), color='r', lw=2)  # indicate the avg waveform
    # ax.plot(x_time, np.nanmedian(spk_waveform, axis=0), color='r', lw=2)  # indicate the median waveform
    plt.xlim([-0.05, 1])
    plt.title(cell_name)
    plt.show()

    # Waveform analysis (based on averaged waveform)
    avg_waveform = np.nanmean(spk_waveform, axis=0)
    spk_height = np.abs(np.max(avg_waveform) - np.min(avg_waveform))  # in microseconds
    spk_width = ((np.argmax(avg_waveform) - np.argmin(avg_waveform)) + 1) * (1 / sample_rate) * 1E6  # in microseconds

    # Calculate the SNR (signal-to-noise ratio in dB)
    # variance of the signal (waveform) divided by the total neural trace
    snr = 10 * np.log10(np.var(avg_waveform) / np.var(raw_trace))  # in dB

    # Create a folder to store output files
    dir_name = 'WaveformAnalysis'
    save.make_save_dir(dir_name)
