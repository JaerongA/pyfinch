"""
Compare motif firing rates between different conditions
Get values from unit_profile table
Run firing_rates.py to calculate motif firing rates
"""


from database.load import ProjectLoader
from deafening.results.plot import plot_bar_comparison
import matplotlib.pyplot as plt
from util import save


# Load database
db = ProjectLoader().load_db()
# SQL statement
query = """SELECT unit.*, cluster.taskSessionDeafening, cluster.taskSessionPostDeafening, cluster.dph, cluster.block10days
    FROM unit_profile unit INNER JOIN cluster ON cluster.id = unit.clusterID"""

df = db.to_dataframe(query)
# df.set_index('id')
df.dropna(subset=['motifFRUndir'], inplace=True)  # Drop out NaNs


# # Compare firing rates pre vs. post-deafening
# # Parameters
# nb_row = 3
# nb_col = 3
# save_fig = False
# fig_ext = '.png'
#
# # Plot the results
# fig, ax = plt.subplots(figsize=(9, 4))
# plt.suptitle('Firing Rates', y=.9, fontsize=20)
#
# # Baseline FR
# ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['baselineFR'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Baseline', ylabel='Firing Rates (Hz)',
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Undir
# ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['motifFRUndir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Undir',
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Dir
# ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['motifFRDir'], df['taskName'],
#                     hue_var=df['birdID'],
#                     title='Dir',
#                     col_order=("Predeafening", "Postdeafening"),
#                     legend_ok=True
#                     )
# fig.tight_layout()
#
# # Save results
# if save_fig:
#     save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
#     save.save_fig(fig, save_path, 'Firing Rates', fig_ext=fig_ext)
# else:
#     plt.show()



import seaborn as sns
from util.draw import remove_right_top

# Plot the results
x = df['block10days']
y = df['motifFRUndir']

title = 'Motif FR per day block'
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
plt.suptitle(title, y=1, fontsize=12)

sns.barplot(x, y, ax=axes[0], facecolor=(1, 1, 1, 0),
                 linewidth=1,
                 errcolor=".2", edgecolor=".2", zorder=0)
remove_right_top(axes[0])

sns.violinplot(x, y, ax=axes[1], inner=None)
sns.swarmplot(x, y, ax=axes[1], color="k")
remove_right_top(axes[1])

axes[0].set_xlabel('Day Block (10 days)'), axes[0].set_ylabel('FR (Hz)')
axes[1].set_xlabel('Day Block (10 days)'), axes[1].set_ylabel('')
axes[0].set_ylim([0, 70]), axes[1].set_ylim([0, 70])
day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day 21-30', 'Day >= 31' ]
axes[0].set_xticklabels(day_block_label_list, rotation = 45)
axes[1].set_xticklabels(day_block_label_list, rotation = 45)

# Run one-way ANOVA
import scipy.stats as stats
f_val, p_val = stats.f_oneway(df['motifFRUndir'][df['block10days'] == 0],
                              df['motifFRUndir'][df['block10days'] == 1],
                              df['motifFRUndir'][df['block10days'] == 2],
                              df['motifFRUndir'][df['block10days'] == 3],
                              df['motifFRUndir'][df['block10days'] == 4]
                              )

msg = f"""One-way ANOVA \n\n F={f_val: 0.3f}, p={p_val: 0.3f}"""
axes[2].text(0, 0.5, msg), axes[2].axis('off')

plt.tight_layout()
plt.show()
