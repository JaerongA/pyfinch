"""
By Jaerong
Calculates a analysis signal-to-noise ratio (SNR) relative to the background (raw neural trace)
"""
from analysis.functions import get_snr
from analysis.parameters import *
from analysis.spike import *
from util import save
from util.draw import *
from database.load import ProjectLoader, DBInfo


def plot_waveform(axis, wf_ts, spk_wf,
                  wf_ts_interp=None,
                  avg_wf_interp=None,
                  spk_proportion=0.1,
                  deflection_range=None,
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
        if wf_ts_interp.any() and avg_wf_interp.any():
            axis.plot(wf_ts_interp, avg_wf_interp, color='r', lw=2)  # indicate the avg waveform
        else:
            axis.plot(wf_ts, np.nanmean(spk_wf, axis=0), color='r', lw=2)  # indicate the avg waveform

    axis.set_xlabel('Time (ms)')
    axis.set_ylabel('Amplitude (µV)')
    plt.xlim([-0.2, 1])

    if bool(deflection_range):
        for ind in deflection_range:
            axis.axvline(x=wf_ts_interp[ind], color='r', linewidth=1, ls='--')

    if scale_bar:
        # Plot a scale bar
        plt.plot([-0.1, -0.1], [-250, 250], 'k', lw=2)  # for amplitude
        plt.text(-0.25, -120, '500 µV', rotation=90)
        plt.plot([0, 0.5], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k', lw=2)  # for time
        plt.text(0.15, ax.get_ylim()[0] * 1.05, '500 µs')
        plt.axis('off')


# Parameters
save_fig = True
update_db = True
update = True
save_wf_values = True
dir_name = 'WaveformAnalysis'
fig_ext = '.png'  # .png or .pdf

# Load database
db = ProjectLoader().load_db()

# SQL statement
#query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 4 "
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format

    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    ci.analyze_waveform()  # get waveform features
    nd = NeuralData(path, channel_nb, format, update=update)  # raw neural data

    # Calculate the SNR (signal-to-noise ratio in dB)
    # variance of the signal (waveform) divided by the variance of the total neural trace
    snr = get_snr(ci.avg_wf, nd.data)

    # Plot the individual waveforms
    fig = plt.figure(figsize=(7, 5))
    fig.suptitle(ci.name)
    ax = plt.subplot(121)
    plot_waveform(ax, ci.wf_ts, ci.spk_wf,
                  ci.wf_ts_interp, ci.avg_wf_interp,
                  spk_proportion,
                  ci.deflection_range  # demarcate the deflection point
                  )

    # Print out text
    plt.subplot(122)
    plt.axis('off')
    plt.text(0.1, 0.8, 'SNR = {} dB'.format(snr), fontsize=12)
    plt.text(0.1, 0.6, 'Spk Height = {:.2f} µV'.format(ci.spk_height), fontsize=12)
    plt.text(0.1, 0.4, 'Spk Width = {:.2f} µs'.format(ci.spk_width), fontsize=12)
    plt.text(0.1, 0.2, 'Half width = {:.2f} µs'.format(ci.half_width), fontsize=12)  # measured from the peak deflection
    plt.text(0.1, 0.0, '# of Spk = {}'.format(ci.nb_spk), fontsize=12)
    set_fig_size(4.2, 2.5)  # set the physical size of the save_fig in inches (width, height)

    # Save results to database
    if update_db:
        db.create_col('cluster', 'nbSpk', 'INT')
        db.update('cluster', 'nbSpk', row['id'], ci.nb_spk)
        db.create_col('cluster', 'SNR', 'REAL')
        db.update('cluster', 'SNR', row['id'], snr)
        db.create_col('cluster', 'spkHeight', 'REAL')
        db.update('cluster', 'spkHeight', row['id'], ci.spk_height)
        db.create_col('cluster', 'spkWidth', 'REAL')
        db.update('cluster', 'spkWidth', row['id'], ci.spk_width)
        db.create_col('cluster', 'spkHalfWidth', 'REAL')
        db.update('cluster', 'spkHalfWidth', row['id'], ci.half_width)

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', dir_name)
        save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)

    else:
        plt.show()
    # Print avg_spk_wf to csv
    if save_wf_values:
        import pandas as pd
        df = pd.DataFrame(ci.avg_wf_interp)
        # csv_path = C:\Users\dreill03\Box\AreaX\Analysis\WaveformAnalysis\AverageWaveforms
        # df.to_csv(ProjectLoader().path / dir_name_2 + ci.name + 'ave_wf.csv', index=False)
        df.to_csv(r'C:\Users\dreill03\Box\AreaX\Analysis\WaveformAnalysis\AverageWaveforms\ave_wf.csv')

#        save_path = save.make_dir(ProjectLoader().path / dir_name_2)
#        saveDKR.save_csv(save_path, ci.name, csv_ext='.csv', open_folder=True)
    else:
        print(ci.avg_wf_interp)


# Convert db to csv
if update_db:
    db.to_csv('cluster')
print('Done!')


df = db.to_dataframe("SELECT DISTINCT birdID, taskName FROM cluster")