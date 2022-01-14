"""
Analyze spike waveform metrics
Calculates signal-to-noise ratio (SNR) relative to the background (raw neural trace)
Save results to unit_profile table
"""


def plot_waveform(ax, wf_ts, spk_wf,
                  wf_ts_interp=None,
                  avg_wf_interp=None,
                  spk_proportion=0.2,
                  deflection_points=None,
                  avg_wf=True,
                  scale_bar=True
                  ):

    """
    Plot individual & avg waveforms
    Parameters
    ----------
    ax
    wf_ts
    spk_wf
    wf_ts_interp
    avg_wf_interp
    spk_proportion
    deflection_points : list
        index of deflection point of a waveform
    avg_wf : bool
        overlay averaged waveform
    scale_bar : bool
        plot the scale bar

    """
    import numpy as np
    from util.draw import remove_right_top

    # Randomly select proportions of waveforms to plot
    np.random.seed(seed=42)
    wf_to_plot = spk_wf[np.random.choice(spk_wf.shape[0], size=int(spk_wf.shape[0] * spk_proportion), replace=False)]

    for wf in wf_to_plot:
        ax.plot(wf_ts, wf, color='k', lw=0.2)

    remove_right_top(ax)

    if avg_wf:
        if wf_ts_interp and avg_wf_interp:
            ax.plot(wf_ts_interp, avg_wf_interp, color='r', lw=2)  # indicate the avg waveform
        else:
            ax.plot(wf_ts, np.nanmean(spk_wf, axis=0), color='r', lw=2)  # indicate the avg waveform

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_xlim([-0.2, 1])

    if bool(deflection_points):
        for ind in deflection_points:
            if wf_ts_interp:
                ax.axvline(x=wf_ts_interp[ind], color='r', linewidth=1, ls='--')
            else:
                ax.axvline(x=wf_ts[ind], color='r', linewidth=1, ls='--')

    if scale_bar:
        # Plot a scale bar
        ax.plot([-0.1, -0.1], [-250, 250], 'k', lw=2)  # for amplitude
        ax.text(-0.25, -120, '500 µV', rotation=90)
        ax.plot([0, 0.5], [ax.get_ylim()[0], ax.get_ylim()[0]], 'k', lw=2)  # for time
        ax.text(0.15, ax.get_ylim()[0] * 1.05, '500 µs')
        ax.axis('off')


def main(query,
         spk_proportion=0.5,
         interpolate=True,
         align_wf=True,
         filter_crit=5,
         update=True,
         save_fig=True,
         update_db=True,
         save_folder_name='Waveform',
         view_folder=True,
         fig_ext='.png'
         ):

    from analysis.functions import get_snr
    from analysis.spike import ClusterInfo, NeuralData
    from database.load import ProjectLoader, DBInfo
    import matplotlib.pyplot as plt
    from util import save
    from util.draw import set_fig_size

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
        ci.analyze_waveform(interpolate=True, interp_factor=100, align_wf=align_wf)  # get waveform features
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
                      spk_proportion=spk_proportion
                      )

        # Print out text
        plt.subplot(122)
        plt.axis('off')
        plt.text(0.1, 0.8, 'SNR = {} dB'.format(snr), fontsize=font_size)
        plt.text(0.1, 0.6, 'Spk Height = {:.2f} µV'.format(ci.spk_height), fontsize=font_size)
        plt.text(0.1, 0.4, 'Spk Width = {:.2f} µs'.format(ci.spk_width), fontsize=font_size)
        plt.text(0.1, 0.2, '# of Spk = {}'.format(ci.nb_spk), fontsize=font_size)
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

    from database.load import create_db

    # Parameters
    spk_proportion = 0.5  # proportion of waveforms to plot (0 to 1)
    interpolate = True  # interpolate spike waveform to artificially increase the number of data points
    align_wf = True  # align waveform first to calculate waveform metrics
    filter_crit = 5  # neural data envelope threshold (in s.d), filter out values exceeding this threshold
    update = False
    save_fig = True
    update_db = True
    view_folder = True
    fig_ext = '.png'  # .png or .pdf

    # SQL statement
    query = "SELECT * FROM cluster"

    main(query,
         spk_proportion=spk_proportion,
         interpolate=interpolate,
         align_wf=align_wf,
         filter_crit=filter_crit,
         update=update,
         save_fig=save_fig,
         update_db=update_db,
         view_folder=view_folder,
         fig_ext=fig_ext
         )
