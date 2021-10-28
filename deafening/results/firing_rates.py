"""
Compare motif firing rates between different conditions
Get values from unit_profile table
Run firing_rates.py to calculate motif firing rates
"""

from database.load import ProjectLoader
from deafening.plot import plot_bar_comparison, plot_per_day_block
import matplotlib.pyplot as plt
from util import save

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = """SELECT unit.*, cluster.taskSessionDeafening, cluster.taskSessionPostDeafening, cluster.dph, cluster.block10days
    FROM unit_profile unit INNER JOIN cluster ON cluster.id = unit.clusterID"""

df = db.to_dataframe(query)
df.dropna(subset=['motifFRUndir'], inplace=True)  # Drop out NaNs

# Compare firing rates pre vs. post-deafening
# Parameters
nb_row = 3
nb_col = 3
save_fig = False
fig_ext = '.png'

# Plot the results
fig, ax = plt.subplots(figsize=(9, 4))
plt.suptitle('Firing Rates', y=.9, fontsize=20)

# Baseline FR
ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['baselineFR'], df['taskName'],
                    hue_var=df['birdID'],
                    title='Baseline', ylabel='Firing Rates (Hz)',
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Undir
ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['motifFRUndir'], df['taskName'],
                    hue_var=df['birdID'],
                    title='Undir',
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Dir
ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['motifFRDir'], df['taskName'],
                    hue_var=df['birdID'],
                    title='Dir',
                    col_order=("Predeafening", "Postdeafening"),
                    legend_ok=True
                    )
fig.tight_layout()

# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'Firing Rates', fig_ext=fig_ext)
else:
    plt.show()

# Plot motif FR per day blocks
plot_per_day_block(df, ind_var_name='block10days', dep_var_name='motifFRUndir',
                   title='Motif FR (Undir) per day block',
                   y_label='FR (Hz)', y_lim=[0, 70],
                   view_folder=True,
                   fig_name='MotifFR_per_day_block',
                   save_fig=False, fig_ext='.png'
                   )

from deafening.plot import plot_paired_data
import numpy as np
from scipy import stats
import seaborn as sns
from util.draw import remove_right_top

# plot_paired_data(df, x='taskName', y='entropyUndir',
#                      x_label=None, y_label="Spectral Entropy",
#                      y_lim=[0, 1.2],
#                      view_folder=True,
#                      fig_name='Spectral_entropy_comparison',
#                      save_fig=False,
#                      save_path=save_path,
#                      fig_ext='.png'
#                      )

# Make paired-comparisons between baseline and motif firing rates
fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=150)
plt.suptitle('Motif FR (Undir)', y=.98, fontsize=10)
dot_size = 20

tasks = sorted(df['taskName'].unique(), reverse=True)

# Plot scatter
for ind, task in enumerate(tasks):

    df_task = df.query(f"taskName == '{task}'")
    sns.scatterplot(np.zeros(len(df_task['baselineFR'])), df_task['baselineFR'],
                    s=dot_size, color='k', ax=ax[ind])

    sns.scatterplot(np.ones(len(df_task['motifFRUndir'])), df_task['motifFRUndir'],
                    s=dot_size,  color='k', ax=ax[ind])
    remove_right_top(ax[ind])

    # Plot connecting lines
    for cluster in range(len(df_task['baselineFR'])):
        ax[ind].plot([0, 1],
                     [df_task['baselineFR'], df_task['motifFRUndir']],
                     'k-', linewidth=0.5)

    # 2 sample paired t-test
    group1, group2 = df_task['baselineFR'], df_task['motifFRUndir']
    tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
    degree_of_freedom = len(group1) + len(group2) - 2
    msg1 = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
    if pval < 0.001:  # mark significance
        msg2 = ('p < 0.001')
    else:
        msg2 = ('p = {:.3f}'.format(pval))
    msg = msg1 + ', ' + msg2

    # rank-sum
    # group1, group2 = [], []
    # group1 = df_mean.query('taskName == "Predeafening"')['entropyUndir']
    # group2 = df_mean.query('taskName == "Postdeafening"')['entropyUndir']
    # stat, pval = stats.ranksums(group1, group2)
    # degree_of_freedom = len(group1) + len(group2) - 2
    # msg = f"ranksum p-val = {pval : .3f}"

    # if pval < 0.001:
    #     sig = '***'
    # elif pval < 0.01:
    #     sig = '**'
    # elif pval < 0.05:
    #     sig = '*'
    # else:
    #     sig = 'ns'
    #
    # x1, x2 = 0, 1
    # y_loc, h, col = max(group1.max(), group2.max()) + 5, 0.05, 'k'
    # ax[ind].plot([x1, x1, x2, x2], [y_loc, y_loc + h, y_loc + h, y_loc], lw=1, c=col)
    # ax[ind].text((x1 + x2) * .5, y_loc + h * 1.5, sig, ha='center', va='bottom', color=col, size=10)

    ax[ind].set_ylim([-5, 60])
    ax[ind].set_xlim([-0.1, 1.1])
    ax[ind].set_xticks([0, 1])
    ax[ind].set_xticklabels(['Baseline', 'Motif FR'])
    if ind == 0:
        ax[ind].set_title(f"Predeafening (n={len(df_task)}) \n\n {msg}", size=8)
        ax[ind].set_ylabel('FR')
    else:
        ax[ind].set_title(f"Postdeafening (n={len(df_task)}) \n\n {msg}", size=8)
        ax[ind].set_ylabel('')

plt.show()

