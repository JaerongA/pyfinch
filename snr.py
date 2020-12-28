"""
By Jaerong
Calculates a analysis signal-to-noise ratio (SNR) relative to the background (raw neural trace)
"""

from database import load
import numpy as np
import scipy.io
from analysis.parameters import sample_rate
from analysis.load import read_spk_txt
import matplotlib.pyplot as plt
from util import draw, save

# Load from the database

# query = "SELECT * FROM cluster WHERE id = 29"
# query = "SELECT * FROM cluster WHERE ephysOK = 1 AND id == 12"
query = "SELECT * FROM cluster WHERE ephysOK = 1"
# query = "SELECT * FROM cluster WHERE id BETWEEN 25 AND 28"

cur, conn, col_names = load.database(query)

for row in cur.fetchall():
    cell_name, cell_path = load.cluster_info(row)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + row['channel'] + '(merged).mat'))[0]
    channel_info = scipy.io.loadmat(mat_file)
    unit_nb = int(row['unit'][-2:])
    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    # Extract the raw neural trace (from the .mat file)
    raw_trace = channel_info['amplifier_data'][0]

    # Read from the cluster .txt file
    spk_ts, spk_wf, nb_spk = read_spk_txt(spk_txt_file, unit_nb)

    # Waveform analysis (based on averaged waveform)
    avg_wf = np.nanmean(spk_wf, axis=0)
    spk_height = np.abs(np.max(avg_wf) - np.min(avg_wf))  # in microseconds
    spk_width = abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1)) * (
            1 / sample_rate) * 1E6  # in microseconds

    # Calculate the SNR (signal-to-noise ratio in dB)
    # variance of the signal (waveform) divided by the total neural trace
    snr = 10 * np.log10(np.var(avg_wf) / np.var(raw_trace))  # in dB

    # Plot the individual waveforms
    fig = plt.figure()
    fig.suptitle(cell_name)
    ax = plt.subplot(121)
    x_time = np.arange(0, spk_wf.shape[1]) / sample_rate * 1E3  # x-axis in ms

    for wave in spk_wf:
        ax.plot(x_time, wave, color='k', lw=0.2)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.plot(x_time, np.nanmean(spk_wf, axis=0), color='r', lw=2)  # indicate the avg waveform
    # ax.plot(x_time, np.nanmedian(spk_waveform, axis=0), color='r', lw=2)  # indicate the median waveform
    plt.xlim([-0.2, 1])

    # Plot a scale bar

    plt.plot([-0.1, -0.1], [-250, 250], 'k', lw=2)  # for amplitude
    plt.text(-0.25, -120, '500 µV', rotation=90)
    plt.plot([0, 0.5], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k', lw=2)  # for time
    plt.text(0.15, ax.get_ylim()[0] * 1.05, '500 µs')
    plt.axis('off')

    # Print out text
    plt.subplot(122)
    plt.axis('off')
    plt.text(0.1, 0.1, 'SNR = {:.2f} dB'.format(snr), fontsize=12)
    plt.text(0.1, 0.3, 'Spk Height = {:.2f} µV'.format(spk_height), fontsize=12)
    plt.text(0.1, 0.5, 'Spk Width = {:.2f} µs'.format(spk_width), fontsize=12)
    plt.text(0.1, 0.7, '# of Spk = {}'.format(nb_spk), fontsize=12)
    draw.set_fig_size(4.2, 2.5)  # set the physical size of the save_fig in inches (width, height)

    # Create a folder to store output files

    dir_name = 'WaveformAnalysis'
    save_dir = save.make_save_dir(dir_name)

    # Save save_fig (.pdf or .png)
    # save.save_fig(fig, save_dir, cell_name, ext='.pdf')  # in vector format
    save.save_fig(fig, save_dir, cell_name, ext='.png')
    # plt.show()
    plt.close(fig)

print('Done!')