"""Compare spike fano factor between different conditions"""


def plot_fano_factor_cluster(save_fig=True,
                             view_folder=False,
                             fig_ext='.png'):
    import numpy as np
    from util import save

    # # SQL statement
    # Fano factor at the motif level
    df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
    df.set_index('id')

    # Parameters
    nb_row = 3
    nb_col = 2

    # Plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.suptitle('Fano Factor', y=.9, fontsize=20)

    # Undir
    df['fanoSpkCountUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
    ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['fanoSpkCountUndir'], df['taskName'], hue_var=df['birdID'],
                        title='Undir', ylabel='Fano Factor',
                        y_lim=[0, round(df['fanoSpkCountUndir'].max()) + 1],
                        col_order=("Predeafening", "Postdeafening")
                        )

    # Dir
    df['fanoSpkCountDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
    ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['fanoSpkCountDir'], df['taskName'], hue_var=df['birdID'],
                        title='Dir',
                        y_lim=[0, round(df['fanoSpkCountDir'].max()) + 1],
                        col_order=("Predeafening", "Postdeafening")
                        )
    fig.tight_layout()

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, 'FanoFactor', fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()


def plot_fano_factor_syllable(save_fig=True,
                              view_folder=False,
                              fig_ext='.png'):
    # Parameters
    nb_row = 3
    nb_col = 2

    # Fano factor (spike counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(f"Fano Factor (FR >= {fr_crit} # of Notes >= {nb_note_crit})", y=.9, fontsize=20)

    # Undir
    query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND frUndir >= {fr_crit}"
    df = db.to_dataframe(query)
    ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['fanoFactorUndir'], df['taskName'],
                        hue_var=df['birdID'],
                        title='Undir', ylabel='Fano factor',
                        y_lim=[0, 6],
                        col_order=("Predeafening", "Postdeafening"),
                        )

    # Dir
    query = f"SELECT * FROM syllable_pcc WHERE nbNoteDir >= {nb_note_crit} AND frDir >= {fr_crit}"
    df = db.to_dataframe(query)
    ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['fanoFactorDir'], df['taskName'],
                        hue_var=df['birdID'],
                        title='Dir',
                        y_lim=[0, round(df['fanoFactorDir'].max() * 10) / 10 + 0.2],
                        col_order=("Predeafening", "Postdeafening"),
                        legend_ok=True
                        )
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    from database.load import ProjectLoader
    from results.plot import plot_bar_comparison
    import matplotlib.pyplot as plt

    save_fig = False
    view_folder = True  # open the folder where the result figures are saved
    fig_ext = '.png'

    nb_note_crit = 10
    fr_crit = 10

    # Load database
    db = ProjectLoader().load_db()

    # plot_fano_factor_cluster(
    #     save_fig=save_fig,
    #     view_folder=view_folder,
    #     fig_ext=fig_ext
    # )


    # plot_fano_factor_syllable(
    #     save_fig=save_fig,
    #     view_folder=view_folder,
    #     fig_ext=fig_ext
    # )

    import numpy as np
    import seaborn as sns
    from util.draw import remove_right_top

    # Parameters
    nb_row = 3
    nb_col = 2

    # Plot scatter with diagonal
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(f"Fano Factor (FR >= {fr_crit} # of Notes >= {nb_note_crit})", y=.9, fontsize=20)

    db.execute(f"SELECT DISTINCT(taskName) FROM syllable_pcc ORDER BY taskName DESC")
    task_list = [data[0] for data in db.cur.fetchall()]
    
    for ind, task in enumerate(task_list):

        # Load database
        if task =='Predeafening':
            query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND frUndir >= {fr_crit}"
        else:
            query = f"SELECT * FROM syllable_pcc WHERE nbNoteDir >= {nb_note_crit} AND frUnDir >= {fr_crit}"

        df = db.to_dataframe(query)

        df_temp = df[df['taskName'] == task]
        ax = plt.subplot2grid((nb_row, nb_col), (1, ind), rowspan=2, colspan=1)
        sns.scatterplot(ax=ax, x='fanoFactorDir', y='fanoFactorUndir', data=df_temp, size=2, color='k')
        ax.plot([0, 1], [0, 1], 'm--', transform=ax.transAxes, linewidth=1)
        remove_right_top(ax)
        ax.set_aspect('equal')
        ax.set_xlabel('Dir'), ax.set_ylabel('Undir'), ax.set_title(task)
        ax.set_xlim([0, 4]), ax.set_ylim([0, 4])
        ax.get_legend().remove()
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1]+1, 1))
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1]+1, 1))
    plt.show()

