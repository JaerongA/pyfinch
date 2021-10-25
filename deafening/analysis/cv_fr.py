"""Compare CV of firing rates over time"""


from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.plot import plot_bar_comparison
from util import save
from deafening.plot import plot_paired_scatter, plot_regression

# Parameters
nb_row = 3
nb_col = 2

nb_note_crit = 10
fr_crit = 10

save_fig = False
view_folder = True  # open the folder where the result figures are saved
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()

# # CV of firing rates (over time)
# fig, ax = plt.subplots(figsize=(7, 4))
# plt.suptitle(f"CV of Firing Rates (FR >= {fr_crit} # of Notes >= {nb_note_crit})", y=.9, fontsize=20)
#
# # Undir
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND frUndir >= {fr_crit}"
# df = db.to_dataframe(query)
# ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['cvFRUndir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Undir', ylabel='CV of FR',
#                     y_lim=[0, 2],
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Dir
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteDir >= {nb_note_crit} AND frDir >= {fr_crit}"
# df = db.to_dataframe(query)
# ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['cvFRDir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Dir',
#                     y_lim=[0, round(df['cvFRDir'].max() * 10) / 10 + 0.2],
#                     col_order=("Predeafening", "Postdeafening"),
#                     legend_ok=True
#                     )
# fig.tight_layout()

# # Save results
# if save_fig:
#     save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
#     save.save_fig(fig, save_path, 'FanoFactor', fig_ext=fig_ext, view_folder=view_folder)
# else:
#     plt.show()


# from deafening.plot import plot_paired_scatter
#
# # Load database
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND " \
#         f"nbNoteDir >= {nb_note_crit} AND " \
#         f"frUndir >= {fr_crit} AND " \
#         f"frDir >= {fr_crit}"
#
# df = db.to_dataframe(query)
#
# # Paired comparison between Undir and Dir
# plot_paired_scatter(df, 'cvFRDir', 'cvFRUndir',
#                     # hue='birdID',
#                     save_folder_name='CV',
#                     x_lim=[0, 3],
#                     y_lim=[0, 3],
#                     x_label='Dir',
#                     y_label='Undir', tick_freq=1,
#                     title=f"CV of FR (FR >= {fr_crit} # of Notes >= {nb_note_crit}) (Paired)",
#                     save_fig=False,
#                     view_folder=False,
#                     fig_ext='.png')


# Plot over the course of days
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
        f"nbNoteUndir >={nb_note_crit}"
df = db.to_dataframe(query)

title = f'Undir FR over {fr_crit} # of Notes >= {nb_note_crit}'
x_label = 'Days from deafening'
y_label = 'CV of FR'
# x_lim = [-30, 40]
y_lim = [-0.05, 3]

x = df['taskSessionDeafening']
y = df['cvFRUndir']

plot_regression(x, y,
                title=title,
                x_label=x_label, y_label=y_label,
                # x_lim=x_lim,
                y_lim=y_lim,
                fr_criteria=fr_crit,
                save_fig=save_fig,
                # regression_fit=True
                )


