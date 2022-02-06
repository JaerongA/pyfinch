"""
Get time-shifted cross-correlation between song and firing rates
Only for Undir
"""

from analysis.parameters import freq_range, peth_parm, note_color
from analysis.spike import MotifInfo, AudioData
from database.load import DBInfo, ProjectLoader, create_db
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from util import save
import seaborn as sns
from util.functions import myround
from util.draw import remove_right_top, get_ax_lim
import warnings
warnings.filterwarnings('ignore')


def get_binary_song(onsets, offsets):
    """
    Get binarized song signal (0 for silence, 1 for song)
    Parameters
    ----------
    onsets : arr
        syllable onsets
    offsets : arr
        syllable offsets
    Returns
    -------
    song_ts : arr
        song timestamp
    binary_song : arr
        binarized song signal
    """
    from math import ceil
    motif_duration = offsets[-1] - onsets[0]
    song_ts = np.arange(-peth_parm['buffer'], ceil(motif_duration) + peth_parm['bin_size'], peth_parm['bin_size'])
    binary_song = np.zeros(len(song_ts))

    # binary_song[np.where(peth_parm['time_bin'] <= mi.durations[0])]
    for onset, offset in zip(onsets, offsets):
        binary_song[np.where((song_ts >= onset) & (song_ts <= offset))] = 1

    return song_ts, binary_song


def get_cross_corr(sig1, sig2, lag_lim=None):
    """Get cross-correlation between two signals"""
    nb_sig = min(len(sig1), len(sig2))
    sig1 = sig1[:nb_sig]
    sig2 = sig2[:nb_sig]

    def xcorr(x, y, normed=True,
              maxlags=10):
        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')
        c = np.correlate(x, y, mode=2)
        if normed:
            c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
        if maxlags is None:
            maxlags = Nx - 1
        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d' % Nx)
        lags = np.arange(-maxlags, maxlags + 1)
        c = c[Nx - 1 - maxlags:Nx + maxlags]
        return lags, c

    (lags, corr) = xcorr(sig1, sig2, maxlags=lag_lim)
    return corr, lags


def get_cross_corr_heatmap(note_onsets, note_offsets, fr_mat):
    """Get cross_correlation heatmap"""
    corr_mat = np.array([], dtype=np.float32)
    nb_motif = len(fr_mat)
    for motif_run in range(nb_motif):
        onsets = note_onsets[motif_run]
        offsets = note_offsets[motif_run]

        onsets = np.asarray(list(map(float, onsets)))
        offsets = np.asarray(list(map(float, offsets)))

        note_dur = offsets - onsets  # syllable duration
        onsets -= onsets[0]  # start from 0
        offsets = onsets + note_dur

        _, binary_song = get_binary_song(onsets, offsets)
        if not fr_mat[motif_run, :].sum():  # skip if fr = 0
            continue
        corr, lags = get_cross_corr(fr_mat[motif_run, :], binary_song, lag_lim=100)
        corr_mat = np.vstack([corr, corr_mat]) if corr_mat.size else corr  # place the first trial at the bottom

    peak_latency = lags[corr_mat.mean(axis=0).argmax()]
    max_cross_corr = corr_mat.mean(axis=0).max()
    return corr_mat, lags, peak_latency, max_cross_corr


def main():
    # parameters
    rec_yloc = 0.05
    text_yloc = 0.5  # text height
    font_size = 10

    # Create a new db to store results
    if update_db:
        db = create_db('create_song_fr_cross_corr.sql')

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
        motif = cluster_db.motif

        # Load class object
        mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name, update=update)  # cluster object
        audio = AudioData(path, update=update)  # audio object

        """
        Plot spectrogram & peri-event histogram (Just the first rendition)
        only select undir songs
        """
        onset_list = [onsets for onsets, context in zip(mi.onsets, mi.contexts) if context == 'U']
        offset_list = [offsets for offsets, context in zip(mi.offsets, mi.contexts) if context == 'U']

        onsets = onset_list[motif_nb]
        offsets = offset_list[motif_nb]

        # Convert from string to array of floats
        onsets = np.asarray(list(map(float, onsets)))
        offsets = np.asarray(list(map(float, offsets)))

        # Motif start and end
        start = onsets[0] - peth_parm['buffer']
        end = offsets[-1] + peth_parm['buffer']
        duration = offsets[-1] - onsets[0]

        # Get spectrogram
        timestamp, data = audio.extract([start, end])
        spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)

        # Plot figure
        fig = plt.figure(figsize=(8, 11))

        fig.set_tight_layout(False)
        fig_name = mi.name
        plt.suptitle(fig_name, y=.93)
        gs = gridspec.GridSpec(20, 8)
        gs.update(wspace=0.025, hspace=0.05)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3, 0:6])
        spect_time = spect_time - spect_time[0] - peth_parm['buffer']  # starts from zero
        ax_spect.pcolormesh(spect_time, spect_freq, spect,
                            cmap='hot_r', rasterized=True,
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.3,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_xlim(-peth_parm['buffer'], duration + peth_parm['buffer'])
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0, 0:6], sharex=ax_spect)
        note_dur = offsets - onsets  # syllable duration
        onsets -= onsets[0]  # start from 0
        offsets = onsets + note_dur

        # Mark syllables
        for i, syl in enumerate(mi.motif):
            rectangle = plt.Rectangle((onsets[i], rec_yloc), note_dur[i], 0.2,
                                      linewidth=1, alpha=0.5, edgecolor='k', facecolor=note_color['Motif'][i])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onsets[i] + (offsets[i] - onsets[i]) / 2), text_yloc, syl, size=font_size)
        ax_syl.axis('off')

        # Plot song amplitude
        ax_amp = plt.subplot(gs[4:6, 0:6], sharex=ax_spect)
        timestamp = timestamp - timestamp[0] - peth_parm['buffer']
        data = stats.zscore(data)
        ax_amp.plot(timestamp, data, 'k', lw=0.1)
        ax_amp.set_ylabel('Amplitude (zscore)', fontsize=font_size)
        ax_amp.set_ylim(-5, 5)
        plt.setp(ax_amp.get_xticklabels(), visible=False)
        ax_amp.set_title(f"Motif = {motif_nb}", fontsize=font_size)
        remove_right_top(ax_amp)

        # Plot binarized song & firing rates
        pi = mi.get_peth()  # peth object
        pi.get_fr(gaussian_std=gaussian_std)  # get firing rates

        # Binarized song signal (0 = silence, 1 = song) Example from the first trial
        song_ts, binary_song = get_binary_song(onsets, offsets)

        ax_song = plt.subplot(gs[7:9, 0:6], sharex=ax_spect)
        ax_song.plot(song_ts, binary_song, color=[0.5, 0.5, 0.5], linewidth=1, ls='--')
        ax_song.set_ylim([0, 1])
        ax_song.set_yticks([])
        ax_song.set_xlabel('Time (ms)', fontsize=font_size)
        ax_song.spines['left'].set_visible(False)
        ax_song.spines['top'].set_visible(False)

        # Plot firing rates on the same axis
        ax_fr = ax_song.twinx()
        ax_fr.plot(pi.time_bin, pi.fr['U'][motif_nb, :], 'k')
        ax_fr.set_ylabel('FR (Hz)', fontsize=font_size)
        fr_ymax = myround(round(ax_fr.get_ylim()[1], 3), base=5)
        ax_fr.set_ylim(0, fr_ymax)
        plt.yticks([0, ax_fr.get_ylim()[1]], [str(0), str(int(fr_ymax))])
        ax_fr.spines['left'].set_visible(False)
        ax_fr.spines['top'].set_visible(False)

        # Plot cross-correlation between binarized song and firing rates
        ax_corr = plt.subplot(gs[10:12, 1:5])
        corr, lags = get_cross_corr(pi.fr['U'][motif_nb, :], binary_song, lag_lim=100)
        ax_corr.plot(lags, corr, 'k')
        ax_corr.set_ylabel('Cross-correlation', fontsize=font_size)
        ax_corr.set_xlabel('Time (ms)', fontsize=font_size)
        ax_corr.set_xlim([-100, 100])
        # ax_corr.axvline(x=lags[corr.argmax()], color='r', linewidth=1, ls='--')  # mark the peak location
        ax_min, ax_max = get_ax_lim(ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], base=10)
        ax_corr.set_yticks([ax_min, ax_max])
        ax_corr.set_yticklabels([ax_min, ax_max])
        ax_corr.set_ylim([ax_min, ax_max])
        remove_right_top(ax_corr)
        del corr, lags

        # Get cross-correlation heatmap across all renditions
        corr_mat, lags, peak_latency, max_cross_corr = get_cross_corr_heatmap(onset_list, offset_list, pi.fr['U'])

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_heatmap = plt.subplot(gs[14:16, 1:5])

        cax = inset_axes(ax_heatmap,
                         width="3%",
                         height="100%",
                         loc='center right',
                         bbox_to_anchor=(0.06, 0., 1, 1),
                         bbox_transform=ax_heatmap.transAxes, borderpad=0
                         )
        cax.set_frame_on(True)

        ax_min, ax_max = get_ax_lim(corr_mat.min(), corr_mat.max(), base=10)
        sns.heatmap(corr_mat,
                    vmin=ax_min, vmax=ax_max,
                    cmap='binary',
                    ax=ax_heatmap,
                    cbar_kws={'ticks': [ax_min, ax_max]},
                    cbar_ax=cax, rasterized=True
                    )
        ax_heatmap.set_title('All renditions')
        ax_heatmap.set_yticks([0, corr_mat.shape[0]])
        ax_heatmap.set_yticklabels([corr_mat.shape[0], 0], rotation=0)
        ax_heatmap.set_ylabel('Renditions')
        ax_heatmap.set_xticks([])
        remove_right_top(ax_heatmap)

        # Get the cross-corr mean across all renditions
        ax_corr_mean = plt.subplot(gs[17:19, 1:5])
        ax_corr_mean.plot(lags, corr_mat.mean(axis=0), 'k')
        ax_corr_mean.set_ylabel('Cross-correlation', fontsize=font_size)
        ax_corr_mean.set_xlabel('Time (ms)', fontsize=font_size)
        ax_corr_mean.set_xlim([-100, 100])
        ax_corr_mean.axvline(x=peak_latency, color='r', linewidth=1, ls='--')  # mark the peak location
        ax_min, ax_max = get_ax_lim(ax_corr_mean.get_ylim()[0], ax_corr_mean.get_ylim()[1], base=10)
        ax_corr_mean.set_yticks([ax_min, ax_max])
        ax_corr_mean.set_yticklabels([ax_min, ax_max])
        ax_corr_mean.set_ylim([ax_min, ax_max])
        remove_right_top(ax_corr_mean)

        # Print out results on the figure
        txt_xloc = 0.2
        txt_yloc = 0.5
        txt_inc = 1  # y-distance between texts within the same section

        ax_txt = plt.subplot(gs[15, 6])
        ax_txt.set_axis_off()  # remove all axes
        ax_txt.text(txt_xloc, txt_yloc,
                    f"# Motifs (Undir) = {len(pi.fr['U'])}",
                    fontsize=font_size)
        txt_yloc -= txt_inc

        ax_txt.text(txt_xloc, txt_yloc,
                    f"Gauss std = {gaussian_std}",
                    fontsize=font_size)
        txt_yloc -= txt_inc

        ax_txt.text(txt_xloc, txt_yloc,
                    f"Cross-corr max = {max_cross_corr : 0.3f}",
                    fontsize=font_size)
        txt_yloc -= txt_inc

        ax_txt.text(txt_xloc, txt_yloc,
                    f"Peak latency = {peak_latency} (ms)",
                    fontsize=font_size)
        txt_yloc -= txt_inc

        # Save results to database
        if update_db:
            db.cur.execute(f"INSERT OR IGNORE INTO song_fr_cross_corr (clusterID) VALUES ({cluster_db.id})")
            db.conn.commit()

            db.cur.execute(f"""UPDATE song_fr_cross_corr 
            SET nbMotifUndir = ({len(pi.fr['U'])}), crossCorrMax = ({max_cross_corr : 0.3f}), peakLatency = ({peak_latency})
            WHERE clusterID = ({cluster_db.id})""")
            db.conn.commit()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder)
        else:
            plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('song_fr_cross_corr')
    print('Done!')


if __name__ == '__main__':

    # Parameters
    update = False  # Update the cache file per cluster
    motif_nb = 24
    gaussian_std = 8  # gaussian kernel for smoothing firing rates
    update_db = False
    save_fig = True
    view_folder = True  # open the folder where the result figures are saved
    save_folder_name = 'SongFR_CrossCorr'
    fig_ext = '.pdf'  # set to '.pdf' for vector output (.png by default)

    # SQL statement
    query = "SELECT * FROM cluster WHERE analysisOK AND id=66"

    main()
