"""
By Jaerong
Inter-spike interval analysis
"""


from analysis.spike import MotifInfo, BaselineInfo
from database.load import ProjectLoader, DBInfo
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


# Parameter
nb_row = 7
nb_col = 3
update = False
save_fig = True
update_db = False  # save results to DB
fig_ext = '.png'  # .png or .pdf

font_size = 12
txt_xloc = 0
txt_yloc = 1
txt_inc = 0.15


# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
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
    bi = BaselineInfo(path, channel_nb, unit_nb, format, name, update=update)  # bout object

    # Get ISI per condition
    isi = mi.get_isi(add_premotor_spk=False)
    isi['B'] = bi.get_isi()

    # Plot the results
    fig = plt.figure(figsize=(13, 7))
    fig.set_dpi(500)
    plt.suptitle(mi.name, y=.95)

    if 'B'in isi.keys():
        ax1 = plt.subplot2grid((nb_row, nb_col), (0, 0), rowspan=3, colspan=1)
        # ax1.plot(spk_corr_parm['time_bin'], isi['B'], 'Baseline', normalize=normalize)
        # ax1.plot(spk_corr_parm['time_bin'], isi['B'], 'Baseline', normalize=normalize)
        # corr_b.plot_corr(ax1, spk_corr_parm['time_bin'], correlogram['B'], 'Baseline', normalize=normalize)
        # ax_txt1 = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=3, colspan=1)
        # print_out_text(ax_txt1, corr_b.peak_latency, bi.mean_fr,
        #                burst_info_b.mean_duration, burst_info_b.freq, burst_info_b.mean_nb_spk,
        #                burst_info_b.fraction, corr_b.category, corr_b.burst_index)

    # if corr_u:
    #     ax2 = plt.subplot2grid((nb_row, nb_col), (0, 1), rowspan=3, colspan=1)
    #     corr_u.plot_corr(ax2, spk_corr_parm['time_bin'], correlogram['U'], 'Undir', normalize=normalize)
    #     ax_txt2 = plt.subplot2grid((nb_row, nb_col), (4, 1), rowspan=3, colspan=1)
    #     print_out_text(ax_txt2, corr_u.peak_latency, mi.mean_fr['U'],
    #                    burst_info_u.mean_duration, burst_info_u.freq, burst_info_u.mean_nb_spk,
    #                    burst_info_u.fraction, corr_u.category, corr_u.burst_index)
    #
    # if corr_d:
    #     ax3 = plt.subplot2grid((nb_row, nb_col), (0, 2), rowspan=3, colspan=1)
    #     corr_d.plot_corr(ax3, spk_corr_parm['time_bin'], correlogram['D'], 'Dir', normalize=normalize)
    #     ax_txt3 = plt.subplot2grid((nb_row, nb_col), (4, 2), rowspan=3, colspan=1)
    #     print_out_text(ax_txt3, corr_d.peak_latency, mi.mean_fr['D'],
    #                    burst_info_d.mean_duration, burst_info_d.freq, burst_info_d.mean_nb_spk,
    #                    burst_info_d.fraction, corr_d.category, corr_d.burst_index)




    # # Save results to database
    # if update_db:
    #     # Baseline
    #     db.create_col('cluster', 'burstDurationBaseline', 'REAL')
    #     db.update('cluster', 'burstDurationBaseline', row['id'], burst_info_b.mean_duration)
    #     db.create_col('cluster', 'burstFreqBaseline', 'REAL')
    #     db.update('cluster', 'burstFreqBaseline', row['id'], burst_info_b.freq)
    #     db.create_col('cluster', 'burstMeanNbSpkBaseline', 'REAL')
    #     db.update('cluster', 'burstMeanNbSpkBaseline', row['id'], burst_info_b.mean_nb_spk)
    #     db.create_col('cluster', 'burstFractionBaseline', 'REAL')
    #     db.update('cluster', 'burstFractionBaseline', row['id'], burst_info_b.fraction)
    #     db.create_col('cluster', 'unitCategoryBaseline', 'STRING')
    #     if corr_b:
    #         db.update('cluster', 'unitCategoryBaseline', row['id'], str(corr_b.category))
    #     db.create_col('cluster', 'burstIndexBaseline', 'REAL')
    #     if corr_b:
    #         db.update('cluster', 'burstIndexBaseline', row['id'], corr_b.burst_index)
    #     # Undir
    #     db.create_col('cluster', 'burstDurationUndir', 'REAL')
    #     db.update('cluster', 'burstDurationUndir', row['id'], burst_info_u.mean_duration)
    #     db.create_col('cluster', 'burstFreqUndir', 'REAL')
    #     db.update('cluster', 'burstFreqUndir', row['id'], burst_info_u.freq)
    #     db.create_col('cluster', 'burstMeanNbSpkUndir', 'REAL')
    #     db.update('cluster', 'burstMeanNbSpkUndir', row['id'], burst_info_u.mean_nb_spk)
    #     db.create_col('cluster', 'burstFractionUndir', 'REAL')
    #     db.update('cluster', 'burstFractionUndir', row['id'], burst_info_u.fraction)
    #     db.create_col('cluster', 'unitCategoryUndir', 'STRING')
    #     if corr_u:
    #         db.update('cluster', 'unitCategoryUndir', row['id'], str(corr_u.category))
    #     db.create_col('cluster', 'burstIndexUndir', 'REAL')
    #     if corr_u:
    #         db.update('cluster', 'burstIndexUndir', row['id'], corr_u.burst_index)
    #     # Dir
    #     db.create_col('cluster', 'burstDurationDir', 'REAL')
    #     db.update('cluster', 'burstDurationDir', row['id'], burst_info_d.mean_duration)
    #     db.create_col('cluster', 'burstFreqDir', 'REAL')
    #     db.update('cluster', 'burstFreqDir', row['id'], burst_info_d.freq)
    #     db.create_col('cluster', 'burstMeanNbSpkDir', 'REAL')
    #     db.update('cluster', 'burstMeanNbSpkDir', row['id'], burst_info_d.mean_nb_spk)
    #     db.create_col('cluster', 'burstFractionDir', 'REAL')
    #     db.update('cluster', 'burstFractionDir', row['id'], burst_info_d.fraction)
    #     db.create_col('cluster', 'unitCategoryDir', 'STRING')
    #     if corr_d:
    #         db.update('cluster', 'unitCategoryDir', row['id'], str(corr_d.category))
    #     db.create_col('cluster', 'burstIndexDir', 'REAL')
    #     if corr_d:
    #         db.update('cluster', 'burstIndexDir', row['id'], corr_d.burst_index)
    #
    #     # Convert db to csv
    #     db.to_csv('cluster')
    #     print('Done!')
    #
    # # Save results
    # if save_fig:
    #     save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'UnitProfiling')
    #     save.save_fig(fig, save_path, mi.name, fig_ext=fig_ext)
    # else:
    #     plt.show()
