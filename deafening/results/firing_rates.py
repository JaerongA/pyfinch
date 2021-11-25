"""
Compare motif firing rates between different conditions
Get values from unit_profile table
Run firing_rates.py to calculate motif firing rates
"""

from database.load import ProjectLoader
from deafening.plot import plot_bar_comparison, plot_per_day_block
import matplotlib.pyplot as plt
import pandas as pd
from util import save

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def compare_baseline_fr(df, x1, x2,
                        x_lim=None, y_lim=None,
                        y_label=None,
                        title=None,
                        fig_name='Untitled',
                        save_fig=False,
                        fig_ext='.png'
                        ):
    import numpy as np
    from scipy.stats import ttest_rel
    import seaborn as sns
    from util.draw import remove_right_top

    """Make paired-comparisons between baseline and motif firing rates"""
    fig, ax = plt.subplots(1, 2, figsize=(5, 3), dpi=150)
    plt.suptitle(title, y=.98, fontsize=10)
    dot_size = 20

    tasks = sorted(df['taskName'].unique(), reverse=True)

    # Plot scatter
    for ind, task in enumerate(tasks):
        df_task = df.query(f"taskName == '{task}'")
        sns.scatterplot(np.zeros(len(df_task[x1])), df_task[x1],
                        s=dot_size, color='k', ax=ax[ind])

        sns.scatterplot(np.ones(len(df_task[x2])), df_task[x2],
                        s=dot_size, color='k', ax=ax[ind])
        remove_right_top(ax[ind])

        # Plot connecting lines
        for cluster in range(len(df_task[x1])):
            ax[ind].plot([0, 1],
                         [df_task[x1], df_task[x2]],
                         'k-', linewidth=0.5)

        # 2 sample paired t-test
        group1, group2 = df_task[x1], df_task[x2]
        stat = ttest_rel(group1, group2)
        degree_of_freedom = len(group1) + len(group2) - 2
        msg1 = f"t({degree_of_freedom}) = {stat.statistic :.2f}"
        if stat.pvalue < 0.001:  # mark significance
            msg2 = "p < 0.001"
        else:
            msg2 = f"p = {stat.pvalue :.3f}"
        msg = msg1 + ', ' + msg2

        ax[ind].set_xlim(x_lim)
        ax[ind].set_xticks([0, 1])
        ax[ind].set_xticklabels(['Baseline', 'Motif FR'])
        ax[ind].set_ylim(y_lim)
        if ind == 0:
            ax[ind].set_title(f"\nPredeafening (n={len(df_task)}) \n\n {msg}", size=8)
            ax[ind].set_ylabel(y_label)
        else:
            ax[ind].set_title(f"\nPostdeafening (n={len(df_task)}) \n\n {msg}", size=8)
            ax[ind].set_ylabel('')
    plt.tight_layout()

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=True)
    else:
        plt.show()


# Load database
db = ProjectLoader().load_db()
# SQL statement
query = """SELECT unit.birdID, unit.taskName, unit.taskSession, 
    unit.baselineFR, unit.motifFRUndir, unit.motifFRDir,  
    cluster.taskSessionDeafening, cluster.taskSessionPostDeafening, cluster.block10days
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
                    title='Baseline', y_label='Firing Rates (Hz)',
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

# Compare motif firing rates relative to baseline firing rates
compare_baseline_fr(df, 'baselineFR', 'motifFRUndir',
                    x_lim=[-0.1, 1.1], y_lim=[-2, 60],
                    y_label='FR',
                    title='Motif FR (Undir)',
                    fig_name='Baseline vs. Motif FR',
                    save_fig=False,
                    fig_ext='.png'
                    )

# Get normalized firing rates
df['motifFRNorm'] = df['motifFRUndir'] / df['baselineFR']

# Plot the results
fig, ax = plt.subplots(figsize=(4, 4))
# plt.suptitle('Firing Rates', y=.9, fontsize=10)
plot_bar_comparison(ax, df['motifFRNorm'], df['taskName'],
                    y_lim=[0, 60],
                    # hue_var=df['birdID'],
                    title='Norm. Motif FR(Undir)', y_label='Norm. Motif FR',
                    col_order=("Predeafening", "Postdeafening"),
                    )
plt.tight_layout()

# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'Norm. Motif FR(Undir)', fig_ext=fig_ext)
else:
    plt.show()
