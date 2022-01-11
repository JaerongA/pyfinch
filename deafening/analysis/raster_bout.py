"""
Plot spectrograms & rasters for demarcated song bout
"""


def plot_raster_bouts(query,
                      bout_nb=None,
                      update=False,
                      save_fig=True,
                      view_folder=True,
                      fig_ext='.png'
                      ):
    from analysis.spike import AudioData, BoutInfo, ClusterInfo, NeuralData
    from analysis.parameters import bout_buffer, freq_range, bout_color
    from database.load import ProjectLoader, DBInfo
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from util import save
    from util.draw import remove_right_top
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')

    # Make save path
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name, add_date=False)

    # Parameters
    font_size = 12  # figure font size
    rec_yloc = 0.05
    rect_height = 0.2
    text_yloc = 0.5  # text height
    nb_row = 13
    nb_col = 1
    tick_length = 1
    tick_width = 1

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        unit_nb = int(cluster_db.unit[-2:])
        channel_nb = int(cluster_db.channel[-2:])
        format = cluster_db.format

        # Load data
        ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
        ci.spk_ts = np.concatenate(ci.spk_ts, axis=0)  # concatenate all spike timestamp into a single array
        bi = BoutInfo(path, channel_nb, unit_nb, cluster_db.songNote, format, name, update=update)  # bout object
        audio = AudioData(path, update=update)  # audio object
        nd = NeuralData(path, channel_nb, format, update=update)  # Load neural raw trace

        # Iterate through song bouts
        list_zip = zip(bi.files, bi.onsets, bi.offsets, bi.syllables, bi.contexts)

        for bout_ind, (file, onsets, offsets, syllables, context) in enumerate(list_zip):

            # If you want to plot a specific bout, specify its number
            # Otherwise, plot them all
            if isinstance(bout_nb, int):
                if bout_ind != bout_nb:
                    continue

            # Convert from string to array of floats
            onsets = np.asarray(list(map(float, onsets)))
            offsets = np.asarray(list(map(float, offsets)))
            # spks = spks - onsets[0]

            # bout start and end
            start = onsets[0] - bout_buffer
            end = offsets[-1] + bout_buffer / 2

            spks = ci.spk_ts[np.where((ci.spk_ts >= start) & (ci.spk_ts <= end))]
            spks = spks - onsets[0]
            # spks = spks - bout_buffer

            # Get spectrogram
            timestamp, data = audio.extract([start, end])
            spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)

            # Plot figure
            fig = plt.figure(figsize=(8, 7))
            fig.tight_layout()
            fig_name = f"{file} - Bout # {bout_ind}"
            print("Processing... " + fig_name)
            fig.suptitle(fig_name, y=0.95)

            # Plot spectrogram
            ax_spect = plt.subplot2grid((nb_row, nb_col), (2, 0), rowspan=2, colspan=1)
            spect_time = spect_time - spect_time[0] - bout_buffer
            # spect_time = spect_time - spect_time[0] - bout_buffer

            ax_spect.pcolormesh(spect_time, spect_freq, spect,
                                cmap='hot_r',
                                norm=colors.SymLogNorm(linthresh=0.05,
                                                       linscale=0.03,
                                                       vmin=0.5,
                                                       vmax=100
                                                       ))

            remove_right_top(ax_spect)
            ax_spect.set_ylim(freq_range[0], freq_range[1])
            ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
            plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
            plt.setp(ax_spect.get_xticklabels(), visible=False)
            plt.xlim([spect_time[0] - 100, spect_time[-1] + 100])

            # Plot syllable duration
            ax_syl = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=1, colspan=1, sharex=ax_spect)
            note_dur = offsets - onsets  # syllable duration
            onsets -= onsets[0]  # start from 0
            offsets = onsets + note_dur

            # Mark syllables
            for i, syl in enumerate(syllables):
                rectangle = plt.Rectangle((onsets[i], rec_yloc), note_dur[i], rect_height,
                                          linewidth=1, alpha=0.5, edgecolor='k', facecolor=bout_color[syl])
                ax_syl.add_patch(rectangle)
                ax_syl.text((onsets[i] + (offsets[i] - onsets[i]) / 2.6), text_yloc, syl, size=font_size)
            ax_syl.axis('off')

            # Plot song amplitude
            data = stats.zscore(data)
            timestamp = timestamp - timestamp[0] - bout_buffer
            ax_amp = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=2, colspan=1, sharex=ax_spect)
            ax_amp.plot(timestamp, data, 'k', lw=0.1)
            ax_amp.axis('off')

            # Plot rasters
            ax_raster = plt.subplot2grid((nb_row, nb_col), (6, 0), rowspan=2, colspan=1, sharex=ax_spect)
            # spks2 = spks - start -peth_parm['buffer'] -peth_parm['buffer']
            ax_raster.eventplot(spks, colors='k', lineoffsets=0.5,
                                linelengths=tick_length, linewidths=tick_width, orientation='horizontal')
            ax_raster.axis('off')

            # Plot raw neural data
            timestamp, data = nd.extract([start, end])  # raw neural data
            timestamp = timestamp - timestamp[0] - bout_buffer
            ax_nd = plt.subplot2grid((nb_row, nb_col), (8, 0), rowspan=2, colspan=1, sharex=ax_spect)
            ax_nd.plot(timestamp, data, 'k', lw=0.5)
            ax_nd.set_xlabel('Time (ms)')
            remove_right_top(ax_nd)

            # Add a scale bar
            plt.plot([ax_nd.get_xlim()[0], ax_nd.get_xlim()[0] + 0],
                     [-250, 250], 'k', lw=3)  # for amplitude
            # plt.text(ax_nd.get_xlim()[0] - (bout_buffer / 2), -200, '500 µV', rotation=90)
            ax_nd.set_ylabel('500 µV', rotation=90)
            plt.subplots_adjust(wspace=0, hspace=0)
            ax_nd.spines['left'].set_visible(False)
            plt.yticks([], [])

            # Save results
            if save_fig:
                save_path2 = save.make_dir(save_path / ci.name, add_date=False)
                save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder)
            else:
                plt.show()

    print('Done!')


if __name__ == '__main__':
    # Parameters
    bout_nb = None  # bout index you want to plot (None by default)
    update = False  # Update the cache file per cluster
    save_fig = True
    view_folder = True  # open the folder where the result figures are saved
    fig_ext = '.png'  # set to '.pdf' for vector output (.png by default)
    save_folder_name = 'RasterBouts'

    # SQL statement
    # query = "SELECT * FROM cluster WHERE analysisOK = 1"
    query = "SELECT * FROM cluster WHERE id = 6"

    plot_raster_bouts(query,
                      bout_nb=bout_nb,
                      update=update,
                      save_fig=save_fig,
                      view_folder=view_folder,
                      fig_ext=fig_ext
                      )
