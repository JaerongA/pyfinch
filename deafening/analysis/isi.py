"""
Inter-spike interval analysis
"""


def print_out_text(ax, peak_latency,
                   ref_prop,
                   cv
                   ):
    txt_xloc = 0
    txt_yloc = 0.8
    txt_inc = 0.3
    font_size = 12

    ax.set_ylim([0, 1])
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"ISI peak latency = {round(peak_latency, 3)} (ms)", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"Within Ref Proportion= {round(ref_prop, 3)} %", fontsize=font_size)
    txt_yloc -= txt_inc
    ax.text(txt_xloc, txt_yloc, f"CV of ISI = {cv}", fontsize=font_size)
    ax.axis('off')

    main(query,
         save_fig=save_fig,
         view_folder=view_folder,
         save_folder_name=save_folder_name,
         fig_ext=fig_ext
         )

def main(query,
         save_fig=True,
         view_folder=True,
         save_folder_name='ISI',
         fig_ext='.png'
         ):
    from analysis.spike import MotifInfo, BaselineInfo
    from database.load import ProjectLoader, DBInfo
    import matplotlib.pyplot as plt
    from util import save

    # Parameter
    nb_row = 6
    nb_col = 3

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
        motif = cluster_db.motif

        mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name)  # cluster object
        bi = BaselineInfo(path, channel_nb, unit_nb, format, name)  # baseline object

        # Get ISI per condition
        isi = mi.get_isi(add_premotor_spk=False)
        isi['B'] = bi.get_isi()

        # Plot the results
        fig = plt.figure(figsize=(11, 5))
        fig.set_dpi(550)
        plt.suptitle(mi.name, y=.95)

        if 'B' in isi.keys():
            ax1 = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=3, colspan=1)
            isi['B'].plot(ax1, 'Log ISI (Baseline)')
            ax_txt1 = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=2, colspan=1)
            print_out_text(ax_txt1, isi['B'].peak_latency, isi['B'].within_ref_prop, isi['B'].cv)

        if 'U' in isi.keys():
            ax2 = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=3, colspan=1)
            isi['U'].plot(ax2, 'Log ISI (Undir)')
            ax_txt2 = plt.subplot2grid((nb_row, nb_col), (4, 1), rowspan=2, colspan=1)
            print_out_text(ax_txt2, isi['U'].peak_latency, isi['U'].within_ref_prop, isi['U'].cv)

        if 'D' in isi.keys():
            ax3 = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=3, colspan=1)
            isi['D'].plot(ax3, 'Log ISI (Dir)')
            ax_txt3 = plt.subplot2grid((nb_row, nb_col), (4, 2), rowspan=2, colspan=1)
            print_out_text(ax_txt3, isi['D'].peak_latency, isi['D'].within_ref_prop, isi['D'].cv)

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
            save.save_fig(fig, save_path, mi.name, view_folder=view_folder, fig_ext=fig_ext)
        else:
            plt.show()


if __name__ == '__main__':

    # Parameters
    save_fig = False
    view_folder = False  # view the folder where figures are stored
    save_folder_name = 'ISI'
    fig_ext = '.png'  # .png or .pdf

    # SQL statement
    query = "SELECT * FROM cluster"

    main(query,
         save_fig=save_fig,
         view_folder=view_folder,
         save_folder_name=save_folder_name,
         fig_ext=fig_ext
         )
