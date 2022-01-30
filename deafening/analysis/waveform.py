"""
Analyze spike waveform metrics
Calculates signal-to-noise ratio (SNR) relative to the background (raw neural trace)
Save results to unit_profile table
"""

from analysis.functions import get_snr
from analysis.spike import ClusterInfo, NeuralData
from database.load import create_db, DBInfo, ProjectLoader
import matplotlib.pyplot as plt
from util import save
from util.draw import set_fig_size


def plot_waveform(ax, wf_ts, spk_wf,
                  wf_ts_interp=None,
                  avg_wf_interp=None,
                  spk_proportion=0.2,
                  deflection_points=None,
                  avg_wf=True,
                  plot_individual_wf=False,
                  plot_std=False,
                  scale_bar=True
                  ):
    """
    Plot individual & avg waveforms
    Parameters
    ----------
    ax : axis object
    wf_ts : np.ndarray
    spk_wf : np.ndarray
    wf_ts_interp : np.ndarray
    avg_wf_interp : np.ndarray
    spk_proportion : float
        proportion of spikes to plot (e.g., set 0.5 for plotting 50% of the total waveforms)
    deflection_points : list
        index of deflection point of a waveform
    avg_wf : bool
        overlay averaged waveform
    plot_individual_wf : bool
        plot individual waveforms
    plot_std : bool
        plot std of the waveform
    scale_bar : bool
        plot the scale bar

    """
    import numpy as np
    from util.draw import remove_right_top

    if plot_individual_wf:
        # Randomly select proportions of waveforms to plot
        np.random.seed(seed=42)
        wf_to_plot = spk_wf[
            np.random.choice(spk_wf.shape[0], size=int(spk_wf.shape[0] * spk_proportion), replace=False)]
        for wf in wf_to_plot:
            ax.plot(wf_ts, wf, color='k', lw=0.2)

    elif plot_std:
        # plot std
        ax.fill_between(wf_ts,
                        np.nanmean(spk_wf, axis=0) - 2 * np.nanstd(spk_wf, axis=0),
                        np.nanmean(spk_wf, axis=0) + 2 * np.nanstd(spk_wf, axis=0),
                        alpha=0.3, facecolor='b'
                        )

    remove_right_top(ax)

    if avg_wf:
        if wf_ts_interp and avg_wf_interp:
            ax.plot(wf_ts_interp, avg_wf_interp, color=wf_color, lw=2)  # indicate the avg waveform
        else:
            ax.plot(wf_ts, np.nanmean(spk_wf, axis=0), color=wf_color, lw=2)  # indicate the avg waveform

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_xlim([-0.2, 1])

    if bool(deflection_points):
        for ind in deflection_points:
            if wf_ts_interp:
                ax.axvline(x=wf_ts_interp[ind], color=wf_color, linewidth=1, ls='--')
            else:
                ax.axvline(x=wf_ts[ind], color=wf_color, linewidth=1, ls='--')

    if scale_bar:
        # Plot a scale bar
        ax.plot([-0.1, -0.1], [-250, 250], 'k', lw=2)  # for amplitude
        ax.text(-0.25, -120, '500 µV', rotation=90)
        ax.plot([0, 0.5], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k', lw=2)  # for time
        ax.text(0.15, ax.get_ylim()[0] * 1.05, '500 µs')
        ax.axis('off')


def analyze_waveform():
    font_size = 10

    # Create a new db to store results
    if update_db:
        create_db('create_unit_profile.sql')

    # Load database
    db = ProjectLoader().load_db()
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
        ci.analyze_waveform(interpolate=interpolate, interp_factor=interp_factor,
                            align_wf=align_wf)  # get waveform features
        nd = NeuralData(path, channel_nb, format, update=False)  # raw neural data

        # Calculate the SNR (signal-to-noise ratio in dB)
        # variance of the signal (waveform) divided by the variance of the total neural trace
        snr = get_snr(ci.avg_wf, nd.data, filter_crit=filter_crit)
        del nd

        # Plot the individual waveforms
        fig = plt.figure(figsize=(7, 5))
        fig.suptitle(ci.name)
        ax = plt.subplot(121)
        plot_waveform(ax, wf_ts=ci.wf_ts, spk_wf=ci.spk_wf,
                      spk_proportion=spk_proportion,
                      plot_individual_wf=plot_individual_wf,
                      plot_std=plot_std,
                      )

        # Print out text
        plt.subplot(122)
        plt.axis('off')
        plt.text(0.1, 0.8, f"SNR = {snr} dB", fontsize=font_size)
        plt.text(0.1, 0.6, f"Spk Height = {ci.spk_height :.2f} µV", fontsize=font_size)
        plt.text(0.1, 0.4, f"Spk Width = {ci.spk_width :.2f} µs", fontsize=font_size)
        plt.text(0.1, 0.2, f"# of Spk = {ci.nb_spk}", fontsize=font_size)
        if plot_individual_wf:
            plt.text(0.1, 0.0, f"Proportion = {int(spk_proportion * 100)} %", fontsize=font_size)
        set_fig_size(4.2, 2.5)  # set the physical size of the save_fig in inches (width, height)

        # Save results to database
        if update_db:
            db.cur.execute(f"UPDATE unit_profile SET nbSpk=({ci.nb_spk}) WHERE clusterID=({cluster_db.id})")
            db.cur.execute(f"UPDATE unit_profile SET spkHeight=({ci.spk_height}) WHERE clusterID=({cluster_db.id})")
            db.cur.execute(f"UPDATE unit_profile SET spkWidth=({ci.spk_width}) WHERE clusterID=({cluster_db.id})")
            db.cur.execute(f"UPDATE unit_profile SET SNR=({snr}) WHERE clusterID=({cluster_db.id})")
            db.conn.commit()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
            save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, view_folder=view_folder)
        else:
            plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('unit_profile')
    print('Done!')


if __name__ == '__main__':
    # Parameters
    spk_proportion = 0.5  # proportion of waveforms to plot (0 to 1)
    interpolate = False  # interpolate spike waveform to artificially increase the number of data points
    interp_factor = 100  # interpolation factor (if 10, increase the sample size by x 10)
    align_wf = True  # align waveform first to calculate waveform metrics
    filter_crit = 5  # neural data envelope threshold (in s.d), filter out values exceeding this threshold
    wf_color = 'k'  # color of the averaged waveform
    plot_individual_wf = False
    plot_std = True  # plot std of the waveform
    update = False  # update/create a cache file
    save_fig = True  # save figure
    update_db = False  # save results to DB
    view_folder = True
    fig_ext = '.pdf'  # .png or .pdf
    save_folder_name = 'Waveform'  # figures saved to analysis/save_folder_name

    # SQL statement
    query = "SELECT * FROM cluster"

    analyze_waveform()
