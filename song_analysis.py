"""
By Jaerong
Song analysis
"""

def analyze_song(query, update_cache=False, update_db=True):

    from analysis.song import SongInfo
    from database.load import ProjectLoader, DBInfo

    # Load database
    db = ProjectLoader().load_db()
    with open('database/create_song_table.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        si = SongInfo(path, name, update=update_cache)  # song object

        nb_files = si.nb_files
        nb_bouts = si.nb_bouts(song_db.songNote)
        nb_motifs = si.nb_motifs(song_db.motif)

        mean_nb_intro_notes = si.mean_nb_intro(song_db.introNotes, song_db.songNote)
        song_call_prop = si.song_call_prop(song_db.calls, song_db.songNote)
        mi = si.get_motif_info(song_db.motif)  # Get motif info
        motif_dur = mi.get_motif_duration()  # Get mean motif duration &  CV per context

        if update_db:
            db.cur.execute("UPDATE song SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
            db.cur.execute("UPDATE song SET nbBoutsUndir=?, nbBoutsDir=? WHERE id=?", (nb_bouts['U'], nb_bouts['D'], song_db.id))
            db.cur.execute("UPDATE song SET nbMotifsUndir=?, nbMotifsDir=? WHERE id=?", (nb_motifs['U'], nb_motifs['D'], song_db.id))
            db.cur.execute("UPDATE song SET meanIntroUndir=?, meanIntroDir=? WHERE id=?", (mean_nb_intro_notes['U'], mean_nb_intro_notes['D'], song_db.id))
            db.cur.execute("UPDATE song SET songCallPropUndir=?, songCallPropDir=? WHERE id=?", (song_call_prop['U'], song_call_prop['D'], song_db.id))
            db.cur.execute("UPDATE song SET motifDurationUndir=?, motifDurationDir=? WHERE id=?", (motif_dur['mean']['U'], motif_dur['mean']['D'], song_db.id))
            db.cur.execute("UPDATE song SET motifDurationCVUndir=?, motifDurationCVDir=? WHERE id=?", (motif_dur['cv']['U'], motif_dur['cv']['D'], song_db.id))
            db.conn.commit()
        else:
            print(nb_files, nb_bouts, nb_motifs, mean_nb_intro_notes, song_call_prop, motif_dur)

    if update_db:
        db.to_csv(f'song')
    print('Done!')


def plot_across_days(x, y,
                     context,
                     nb_bout_crit=10,
                     title=None,
                     x_lim=None,
                     y_lim=None,
                     fig_ext='.png',
                     save_fig=False):

    from database.load import ProjectLoader
    import matplotlib.pyplot as plt
    import seaborn as sns
    from util import save
    from util.draw import remove_right_top

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    df = db.to_dataframe(f"SELECT * FROM song WHERE nbBoutsUndir >= {nb_bout_crit}")
    df.set_index('id')

    # Plot the results
    circ_size = 0.5
    bird_list = df['birdID'].unique()
    fig, axes = plt.subplots(2, 5, figsize=(21, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)
    if title:
        fig.get_axes()[0].annotate(f"{title} (nb of bouts >= {nb_bout_crit}) {context}", (0.5, 0.97),
                                   xycoords='figure fraction',
                                   ha='center',
                                   fontsize=16)
        axes = axes.ravel()

    for bird, ax_ind in zip(bird_list, range(len(bird_list))):

        temp_df = df.loc[df['birdID'] == bird]
        sns.lineplot(x=x, y=y,
                     data=temp_df, marker='o', color='k', mew=circ_size, ax=axes[ax_ind])
        remove_right_top(axes[ax_ind])
        axes[ax_ind].set_title(bird)
        if ax_ind >= 5:
            axes[ax_ind].set_xlabel('Days from deafening')
        else:
            axes[ax_ind].set_xlabel('')

        if (ax_ind == 0) or (ax_ind == 5):
            axes[ax_ind].set_ylabel(title)
        else:
            axes[ax_ind].set_ylabel('')

        axes[ax_ind].set_xlim(x_lim)
        axes[ax_ind].set_ylim(y_lim)
        axes[ax_ind].axvline(x=0, color='k', linestyle='dashed', linewidth=0.5)

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, title, fig_ext=fig_ext, view_folder=False)
    else:
        plt.show()


# def pre_post_comparison(query,
#                         x, y1, y2,
#                         title=None,
#                         run_stats=True,
#                         y_lim=None,
#                         fig_ext='.png',
#                         save_fig=False,
#                         update_cache=False):
#
#     from database.load import ProjectLoader
#     import matplotlib.pyplot as plt
#     from results.plot import plot_bar_comparison
#
#     # Parameters
#     nb_row = 3
#     nb_col = 2
#
#     # Load database
#     db = ProjectLoader().load_db()
#     # # SQL statement
#
#     df = db.to_dataframe(query)
#     # df.set_index('id')
#
#     # Plot the results
#     fig, ax = plt.subplots(figsize=(7, 4))
#     plt.suptitle(title, y=.9, fontsize=20)
#
#     # Undir
#     ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
#     plot_bar_comparison(ax, df[y1], df[x], hue_var=df['birdID'],
#                         title='Undir', ylabel=y1,
#                         col_order=("Predeafening", "Postdeafening"),
#                         y_lim=y_lim,
#                         run_stats=run_stats
#                         )
#     # Dir
#     ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
#     plot_bar_comparison(ax, df[y2], df[x], hue_var=df['birdID'],
#                         title='Dir', ylabel=y2,
#                         col_order=("Predeafening", "Postdeafening"),
#                         y_lim=y_lim,
#                         run_stats=run_stats,
#                         legend_ok=True
#                         )
#     fig.tight_layout()
#     plt.show()


# Parameters
fig_ext = '.png'
save_fig = False
update_db = True
update_cache = False
nb_bout_crit = 10

# SQL statement
# query = "SELECT * FROM song WHERE id<=13"
# query = "SELECT * FROM song"
query = f"SELECT * FROM song WHERE nbBoutsUndir >= {nb_bout_crit}"

# analyze_song(query, update_cache = update_cache, update_db = update_db )

# plot_across_days_per_note('taskSessionDeafening', 'meanIntroUndir', 'Undir',
#                  nb_bout_crit=nb_bout_crit,
#                  title='Mean_nb_intro_notes',
#                  x_lim=[-30, 70],
#                  y_lim=[0, 8],
#                  fig_ext=fig_ext,
#                  save_fig=save_fig)

# plot_across_days_per_note('taskSessionDeafening', 'songCallPropUndir', 'Undir',
#                  nb_bout_crit=nb_bout_crit,
#                  title='Song_Call_Proportions',
#                  x_lim=[-30, 70],
#                  y_lim=[0, 0.6],
#                  fig_ext=fig_ext,
#                  save_fig=save_fig)
#
# plot_across_days_per_note('taskSessionDeafening', 'motifDurationUndir', 'Undir',
#                  nb_bout_crit=nb_bout_crit,
#                  title='Motif Duration (ms)',
#                  x_lim=[-30, 70],
#                  fig_ext=fig_ext,
#                  save_fig=save_fig)

# plot_across_days_per_note('taskSessionDeafening', 'motifDurationCVUndir', 'Undir',
#                  nb_bout_crit=nb_bout_crit,
#                  title='CV of Motif',
#                  x_lim=[-30, 70],
#                  y_lim=[0, 0.04],
#                  fig_ext=fig_ext,
#                  save_fig=save_fig)

# pre_post_comparison(query, 'taskName',
#                     'meanIntroUndir',
#                     'meanIntroDir',
#                     nb_bout_crit=nb_bout_crit,
#                     title="Mean # of intro notes",
#                     run_stats=True,
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)

# pre_post_comparison(query, 'taskName',
#                     'songCallPropUndir',
#                     'songCallPropDir',
#                     nb_bout_crit=nb_bout_crit,
#                     title="Song Call Proportion",
#                     # run_stats=True,
#                     y_lim=[-0.05, 0.75],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)

# pre_post_comparison(query, 'taskName',
#                     'motifDurationUndir',
#                     'motifDurationDir',
#                     nb_bout_crit=nb_bout_crit,
#                     title="Motif Duration (ms)",
#                     run_stats=True,
#                     y_lim=[0, 800],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)

# pre_post_comparison(query, 'taskName',
#                     'motifDurationCVUndir',
#                     'motifDurationCVDir',
#                     nb_bout_crit=nb_bout_crit,
#                     title="CV of Motif",
#                     run_stats=True,
#                     # y_lim=[-0.005, 0.05],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)


# Analyze Entropy & EV
# query = f"SELECT * FROM syllable WHERE nbNoteUndir >= {nb_note_crit}"

# Spectro-Temporal Entropy
# pre_post_comparison(query, 'taskName',
#                     'spectroTemporalEntropyUndir',
#                     'spectroTemporalEntropyDir',
#                     title="Spectro-Temporal Entropy",
#                     run_stats=True,
#                     y_lim=[0, 1],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)
#
# # Spectral Entropy
# pre_post_comparison(query, 'taskName',
#                     'entropyUndir',
#                     'entropyDir',
#                     title="Spectral Entropy",
#                     run_stats=True,
#                     y_lim=[0, 1],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)

# EV
# pre_post_comparison(query, 'taskName',
#                     'entropyVarUndir',
#                     'entropyVarDir',
#                     title="EV",
#                     run_stats=True,
#                     y_lim=[0, 0.05],
#                     fig_ext=fig_ext,
#                     save_fig=save_fig)


