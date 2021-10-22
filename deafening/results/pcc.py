"""Compare pairwise cross-correlation (pcc) between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.results.plot import plot_bar_comparison
import seaborn as sns

from analysis.parameters import fr_crit, nb_note_crit
from util import save
import numpy as np
from deafening.plot import plot_paired_scatter, plot_regression

# Parameters
nb_row = 3
nb_col = 4
save_fig = False
fig_ext = '.png'

def plot_pcc_regression(x, y,
                        x_label, y_label, title,
                        x_lim=None, y_lim=None,
                        fr_criteria=fr_crit, save_fig=save_fig, regression_fit=True):

    # Load database
    db = ProjectLoader().load_db()
    # # SQL statement
    df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
    df.set_index('id')

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.suptitle('Pairwise CC', y=.9, fontsize=20)

    # Undir
    df['pairwiseCorrUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
    ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['pairwiseCorrUndir'], df['taskName'], hue_var=df['birdID'],
                        title='Undir', ylabel='PCC',
                        y_max=round(df['pairwiseCorrUndir'].max() * 10) / 10 + 0.1,
                        col_order=("Predeafening", "Postdeafening"),
                        )

    # Dir
    df['pairwiseCorrDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
    ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df['pairwiseCorrDir'], df['taskName'], hue_var=df['birdID'],
                        title='Dir', y_max=round(df['pairwiseCorrDir'].max() * 10) / 10 + 0.2,
                        col_order=("Predeafening", "Postdeafening"),
                        )
    fig.tight_layout()

    # Undir (paired comparisons)
    pcc_mean_per_condition = df.groupby(['birdID', 'taskName'])['pairwiseCorrUndir'].mean().to_frame()
    pcc_mean_per_condition.reset_index(inplace=True)

    ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)

    ax = sns.pointplot(x='taskName', y='pairwiseCorrUndir', hue='birdID',
                       data=pcc_mean_per_condition,
                       order=["Predeafening", "Postdeafening"],
                       aspect=.5, hue_order=df['birdID'].unique().tolist(), scale=0.7)

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    title = 'Undir (Paired Comparison)'
    title += '\n\n\n'

    plt.title(title)
    plt.xlabel(''), plt.ylabel('')
    plt.ylim(0, 0.3), plt.xlim(-0.5, 1.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, 'PCC', fig_ext=fig_ext)
    else:
        plt.show()

# Syllable PCC plot across days (with regression)
# SQL statement
# Load database
db = ProjectLoader().load_db()
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit}"
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskSessionDeafening <= 0"
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskName='Postdeafening'"
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
#         f"nbNoteUndir >={nb_note_crit} AND " \
#         f"taskName='Postdeafening'"
df = db.to_dataframe(query)
#
# # DPH
# x = df['dph']
# y = df['pccUndir']
#
# title = f'Undir FR over {fr_crit} # of Notes >= {nb_note_crit}'
# x_label = 'Age (dph)'
# y_label = 'PCC'
# # x_lim = [90, 130]
# y_lim = [-0.05, 0.3]
#
# plot_regression(x, y,
#                 title=title,
#                 x_label=x_label, y_label=y_label,
#                 # x_lim=x_lim,
#                 y_lim=y_lim,
#                 fr_criteria=fr_crit,
#                 save_fig=save_fig,
#                 # regression_fit=True
#                 )
#
#
# # Days from deafening
# x = df['taskSessionDeafening']
# y = df['pccUndir']
#
# title = f'Undir FR over {fr_crit} # of Notes >= {nb_note_crit}'
# x_label = 'Days from deafening'
# y_label = 'PCC'
# # x_lim = [0, 35]
# y_lim = [-0.05, 0.25]
#
# plot_regression(x, y,
#                 title=title,
#                 x_label=x_label, y_label=y_label,
#                 # x_lim=x_lim,
#                 y_lim=y_lim,
#                 fr_criteria=fr_crit,
#                 save_fig=save_fig,
#                 # regression_fit={'linear', 'quadratic'}
#                 )

# Paired comparison between Undir and Dir
# Load database
# query = f"SELECT * FROM syllable_pcc WHERE nbNoteUndir >= {nb_note_crit} AND " \
#         f"nbNoteDir >= {nb_note_crit} AND " \
#         f"frUndir >= {fr_crit} AND " \
#         f"frDir >= {fr_crit}"
#
# df = db.to_dataframe(query)
#
# plot_paired_scatter(df, 'pccDir', 'pccUndir',
#                     # hue='birdID',
#                     save_folder_name='pcc',
#                     x_lim=[0, 0.45],
#                     y_lim=[0, 0.45],
#                     x_label='Dir',
#                     y_label='Undir', tick_freq=0.1,
#                     title=f"PCC syllable (FR >= {fr_crit} # of Notes >= {nb_note_crit}) (Paired)",
#                     save_fig=False,
#                     view_folder=False,
#                     fig_ext='.png')


# Plot data across blocks

import seaborn as sns
from util.draw import remove_right_top

# Plot the results
x = df['block10days']
y = df['pccUndir']
# x.replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error

title = 'Syllable PCC per block (Undir)'
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plt.suptitle(title, y=1, fontsize=12)

sns.barplot(x, y, ax=axes[0], facecolor=(1, 1, 1, 0),
                 linewidth=1,
                 errcolor=".2", edgecolor=".2", zorder=0)
remove_right_top(axes[0])

sns.violinplot(x, y, ax=axes[1], inner=None)
sns.swarmplot(x, y, ax=axes[1], color="k")
remove_right_top(axes[1])

axes[0].set_xlabel('Day Block (10 days)'), axes[0].set_ylabel('PCC')
axes[1].set_xlabel('Day Block (10 days)'), axes[1].set_ylabel('')
#axes[0].set_ylim([0, 70]), axes[1].set_ylim([0, 70])
day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day 21-30', 'Day >= 31' ]
axes[0].set_xticklabels(day_block_label_list, rotation = 45)
axes[1].set_xticklabels(day_block_label_list, rotation = 45)

plt.tight_layout()
plt.show()

