"""
By Jaerong
Calculates a analysis signal-to-noise ratio (SNR) relative to the background (raw neural trace)
"""

from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
import matplotlib.pyplot as plt
from util import save
from util.draw import *


def plot_waveform(axis, wf_ts, spk_wf,
                  spk_proportion=0.1,
                  avg_wf=True,
                  scale_bar=True
                  ):
    import numpy as np

    # select the waveforms to plot
    wf_to_plot = spk_wf[np.random.choice(spk_wf.shape[0], size=int(spk_wf.shape[0] * spk_proportion), replace=False)]

    for wf in wf_to_plot:
        axis.plot(wf_ts, wf, color='k', lw=0.2)

    remove_right_top(axis)
    if avg_wf:
        axis.plot(wf_ts, np.nanmean(spk_wf, axis=0), color='r', lw=2)  # indicate the avg waveform
    axis.set_xlabel('Time (ms)')
    axis.set_ylabel('Amplitude (µV)')
    plt.xlim([-0.2, 1])

    if scale_bar:
        # Plot a scale bar
        plt.plot([-0.1, -0.1], [-250, 250], 'k', lw=2)  # for amplitude
        plt.text(-0.25, -120, '500 µV', rotation=90)
        plt.plot([0, 0.5], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k', lw=2)  # for time
        plt.text(0.15, ax.get_ylim()[0] * 1.05, '500 µs')
        plt.axis('off')


# Parameters
spk_proportion = 0.1  # proportion of waveforms to plot
save_fig = True
update_db = False
dir_name = 'WaveformAnalysis'
fig_ext='.png'  # .png or .pdf

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
db = ProjectLoader().load_db()
db.execute(query)

# Loop through neurons
for row in db.cur.fetchall():

    ci = ClusterInfo(row)  # cluster object
    ci.analyze_waveform()  # get waveform features
    nd = NeuralData(row, update=False)  # raw neural data

    # Calculate the SNR (signal-to-noise ratio in dB)
    # variance of the signal (waveform) divided by the total neural trace
    snr = get_snr(ci.avg_wf, nd.data)

    # Plot the individual waveforms
    fig = plt.figure()
    fig.suptitle(ci.name)
    ax = plt.subplot(121)
    plot_waveform(ax, ci.wf_ts, ci.spk_wf, spk_proportion)

    # Print out text
    plt.subplot(122)
    plt.axis('off')
    plt.text(0.1, 0.1, 'SNR = {} dB'.format(snr), fontsize=12)
    plt.text(0.1, 0.3, 'Spk Height = {:.2f} µV'.format(ci.spk_height), fontsize=12)
    plt.text(0.1, 0.5, 'Spk Width = {:.2f} µs'.format(ci.spk_width), fontsize=12)
    plt.text(0.1, 0.7, '# of Spk = {}'.format(ci.nb_spk), fontsize=12)
    set_fig_size(4.2, 2.5)  # set the physical size of the save_fig in inches (width, height)

    # Save results to database
    if update_db:
        db.create_col('cluster', 'SNR', 'REAL')
        db.update('cluster', 'SNR', snr, row['id'])
        db.create_col('cluster', 'spkHeight', 'REAL')
        db.update('cluster', 'spkHeight', ci.spk_height, row['id'])
        db.create_col('cluster', 'spkWidth', 'REAL')
        db.update('cluster', 'spkWidth', ci.spk_width, row['id'])
        db.create_col('cluster', 'nbSpk', 'INT')
        db.update('cluster', 'nbSpk', ci.nb_spk, row['id'])

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', dir_name)
        save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)
    else:
        plt.show()

print('Done!')
