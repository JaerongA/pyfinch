"""
Compare sparseness
"""

from analysis.parameters import fr_crit, nb_note_crit
from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.plot import plot_bar_comparison
from util import save
from deafening.plot import plot_paired_scatter, plot_regression, plot_per_day_block

# Parameters
nb_row = 3
nb_col = 2

save_fig = False
view_folder = True  # open the folder where the result figures are saved
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()

# # Sparseness
# fig, ax = plt.subplots(figsize=(7, 4))
# plt.suptitle(f"Sparseness (FR >= {fr_crit} # of Notes >= {nb_note_crit})", y=.9, fontsize=20)
#
# # Undir
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND frUndir >= {fr_crit}"
# df = db.to_dataframe(query)
# ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['sparsenessUndir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Undir', ylabel='PCC',
#                     y_lim=[0, round(df['sparsenessUndir'].max() * 10) / 10 + 0.1],
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Dir
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteDir >= {nb_note_crit} AND frDir >= {fr_crit}"
# df = db.to_dataframe(query)
# df['sparsenessDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
# ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['sparsenessDir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Dir',
#                     y_lim=[0, round(df['sparsenessDir'].max() * 10) / 10 + 0.2],
#                     col_order=("Predeafening", "Postdeafening"),
#                     legend_ok=True
#                     )
# fig.tight_layout()
# plt.show()


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
# plot_paired_scatter(df, 'sparsenessDir', 'sparsenessUndir',
#                     hue= 'birdID',
#                     save_folder_name='Sparseness',
#                     x_lim=[0, 0.5],
#                     y_lim=[0, 0.5],
#                     x_label='Dir',
#                     y_label='Undir',
#                     title=f"Sparseness per syllable (FR >= {fr_crit} # of Notes >= {nb_note_crit}) (Paired)",
#                     save_fig=False,
#                     view_folder=False,
#                     fig_ext='.png')


# Plot over the course of days
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
        f"nbNoteUndir >={nb_note_crit}"
df = db.to_dataframe(query)

# plot_regression(x=df['taskSessionDeafening'], y=df['sparsenessUndir'],
#                 title=f'Undir FR over {fr_crit} # of Notes >= {nb_note_crit}',
#                 x_label='Days from deafening',
#                 y_label='Sparseness',
#                 # x_lim=x_lim,
#                 y_lim=[-0.05, 0.5],
#                 fr_criteria=fr_crit,
#                 save_fig=save_fig,
#                 # regression_fit=True
#                 )

# Plot fano factor per syllable across blocks
plot_per_day_block(df, ind_var_name='block10days', dep_var_name='sparsenessUndir',
                   title=f'Sparseness (Undir) per day block FR >= {fr_crit} & # of Notes >= {nb_note_crit}',
                   y_label='Sparseness',
                   y_lim=[0, 0.5],
                   fig_name='Sparseness_syllable_per_day_block',
                   save_fig=False, fig_ext='.png'
                   )


