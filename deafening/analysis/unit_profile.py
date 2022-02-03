"""
Get unit profiles such as spike correlograms or burstiness
"""

from analysis.parameters import spk_corr_parm
from analysis.spike import BaselineInfo, BurstingInfo, Correlogram, MotifInfo
from database.load import ProjectLoader, DBInfo, create_db
import matplotlib.pyplot as plt
from util import save


def print_out_text(ax, peak_latency,
                   firing_rates,
                   burst_mean_duration, burst_freq, burst_mean_spk, burst_fraction,
                   category,
                   burst_index
                   ):
    font_size = 12
    txt_xloc = 0
    txt_yloc = 1
    txt_inc = 0.15
    ax.set_ylim([0, 1])
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"Peak Latency = {round(peak_latency, 3)} (ms)", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"Firing Rates = {round(firing_rates, 3)} Hz", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Duration = {round(burst_mean_duration, 3)}', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Freq = {round(burst_freq, 3)} Hz', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst MeanSpk = {round(burst_mean_spk, 3)}', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Fraction = {round(burst_fraction, 3)} (%)', fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, 'Category = {}'.format(category), fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f'Burst Index = {round(burst_index, 3)}', fontsize=font_size)
    ax.axis('off')


def get_unit_profile():
    # Parameter
    nb_row = 8
    nb_col = 3

    # Create & load database
    if update_db:
        create_db('create_unit_profile.sql')

    # Load database
    db = ProjectLoader().load_db()
    with open('../database/create_unit_profile.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

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

        mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name, update=update)  # cluster object
        bi = BaselineInfo(path, channel_nb, unit_nb, cluster_db.songNote, format, name, update=True)  # bout object

        # Calculate firing rates
        mi.get_mean_fr()

        # Get correlogram
        correlogram = mi.get_correlogram(mi.spk_ts, mi.spk_ts)
        correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)  # add baseline correlogram

        corr_b = corr_u = corr_d = None

        # Get correlogram object per condition
        if 'B' in correlogram.keys():
            corr_b = Correlogram(correlogram['B'])  # Baseline
        if 'U' in correlogram.keys():
            corr_u = Correlogram(correlogram['U'])  # Undirected
        if 'D' in correlogram.keys():
            corr_d = Correlogram(correlogram['D'])  # Directed

        # Get jittered correlogram for getting the baseline
        correlogram_jitter = mi.get_jittered_corr()
        correlogram_jitter['B'] = bi.get_jittered_corr()

        for key, value in correlogram_jitter.items():
            if key == 'B':
                corr_b.category(value)
            elif key == 'U':
                corr_u.category(value)
            elif key == 'D':
                corr_d.category(value)

        # Define burst info object
        burst_info_b = BurstingInfo(bi)
        burst_info_u = BurstingInfo(mi, 'U')
        burst_info_d = BurstingInfo(mi, 'D')

        # Plot the results
        fig = plt.figure(figsize=(11, 6))
        fig.set_dpi(dpi)
        plt.suptitle(mi.name, y=.93)

        if corr_b:
            ax1 = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=3, colspan=1)
            corr_b.plot_corr(ax1, spk_corr_parm['time_bin'], correlogram['B'],
                             'Baseline', xlabel='Time (ms)', ylabel='Count', normalize=normalize)
            ax_txt1 = plt.subplot2grid((nb_row, nb_col), (5, 0), rowspan=3, colspan=1)
            print_out_text(ax_txt1, corr_b.peak_latency, bi.mean_fr,
                           burst_info_b.mean_duration, burst_info_b.freq, burst_info_b.mean_nb_spk,
                           burst_info_b.fraction, corr_b.category, corr_b.burst_index)

        if corr_u:
            ax2 = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=3, colspan=1)
            corr_u.plot_corr(ax2, spk_corr_parm['time_bin'], correlogram['U'],
                             'Undir', xlabel='Time (ms)', normalize=normalize)
            ax_txt2 = plt.subplot2grid((nb_row, nb_col), (5, 1), rowspan=3, colspan=1)
            print_out_text(ax_txt2, corr_u.peak_latency, mi.mean_fr['U'],
                           burst_info_u.mean_duration, burst_info_u.freq, burst_info_u.mean_nb_spk,
                           burst_info_u.fraction, corr_u.category, corr_u.burst_index)

        if corr_d:
            ax3 = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=3, colspan=1)
            corr_d.plot_corr(ax3, spk_corr_parm['time_bin'], correlogram['D'],
                             'Dir', xlabel='Time (ms)', normalize=normalize)
            ax_txt3 = plt.subplot2grid((nb_row, nb_col), (5, 2), rowspan=3, colspan=1)
            print_out_text(ax_txt3, corr_d.peak_latency, mi.mean_fr['D'],
                           burst_info_d.mean_duration, burst_info_d.freq, burst_info_d.mean_nb_spk,
                           burst_info_d.fraction, corr_d.category, corr_d.burst_index)

        # Save results to database
        if update_db:
            # Baseline
            if corr_b:
                db.cur.execute(
                    f"UPDATE unit_profile SET burstDurationBaseline = ('{burst_info_b.mean_duration}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFreqBaseline = ('{burst_info_b.freq}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstMeanNbSpkBaseline = ('{burst_info_b.mean_nb_spk}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFractionBaseline = ('{burst_info_b.fraction}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstIndexBaseline = ('{corr_b.burst_index}') WHERE clusterID = ({cluster_db.id})")

            # Undir
            if corr_u:
                db.cur.execute(
                    f"UPDATE unit_profile SET burstDurationUndir = ('{burst_info_u.mean_duration}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFreqUndir = ('{burst_info_u.freq}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstMeanNbSpkUndir = ('{burst_info_u.mean_nb_spk}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFractionUndir = ('{burst_info_u.fraction}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    "UPDATE {} SET {} = ? WHERE {} = ?".format('unit_profile', 'unitCategoryUndir', 'clusterID'),
                    (corr_u.category, cluster_db.id))
                db.cur.execute(
                    f"UPDATE unit_profile SET burstIndexUndir = ('{corr_u.burst_index}') WHERE clusterID = ({cluster_db.id})")

            # Dir
            if corr_d:
                db.cur.execute(
                    f"UPDATE unit_profile SET burstDurationDir = ('{burst_info_d.mean_duration}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFreqDir = ('{burst_info_d.freq}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstMeanNbSpkDir = ('{burst_info_d.mean_nb_spk}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    f"UPDATE unit_profile SET burstFractionDir = ('{burst_info_d.fraction}') WHERE clusterID = ({cluster_db.id})")
                db.cur.execute(
                    "UPDATE {} SET {} = ? WHERE {} = ?".format('unit_profile', 'unitCategoryDir', 'clusterID'),
                    (corr_d.category, cluster_db.id))
                db.cur.execute(
                    f"UPDATE unit_profile SET burstIndexDir = ('{corr_d.burst_index}') WHERE clusterID = ({cluster_db.id})")
            db.conn.commit()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
            save.save_fig(fig, save_path, mi.name, fig_ext=fig_ext, dpi=dpi, view_folder=True)
        else:
            plt.show()

    # Convert db to csv
    if update_db:
        db.to_csv('unit_profile')
        print('Done!')


if __name__ == '__main__':
    # Parameters
    normalize = False  # normalize correlogram
    update = False
    save_fig = False
    update_db = False  # save results to DB
    fig_ext = '.pdf'  # .png or .pdf
    dpi = 500
    save_folder_name = 'UnitProfiling'

    # SQL statement
    query = "SELECT * FROM cluster WHERE id=43"
    # query = "SELECT * FROM cluster WHERE analysisOK=True"

    get_unit_profile()
