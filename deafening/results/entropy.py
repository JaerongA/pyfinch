# Plot values across days
from analysis.parameters import nb_note_crit
from database.load import ProjectLoader
from deafening.plot import plot_across_days_per_note
from util import save

save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Entropy', add_date=False)
# Load database
query = f"""SELECT syl.*, song.taskSession, song.taskSessionDeafening, song.taskSessionPostDeafening, song.dph, song.block10days
FROM syllable syl INNER JOIN song ON syl.songID = song.id WHERE syl.nbNoteUndir >= {nb_note_crit}"""

df = ProjectLoader().load_db().to_dataframe(query)

# # Spectral Entropy
# plot_across_days_per_note(df, x='taskSessionDeafening', y='entropyUndir',
#                           x_label='Days from deafening',
#                           y_label='Spectral Entropy',
#                           title=f"Spectral Entropy (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='Spectral_entropy_across_days',
#                           xlim=[-40, 80], ylim=[0.2, 1],
#                           vline=0,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )
#
# # Spectro-temporal Entropy
# plot_across_days_per_note(df, x='taskSessionDeafening', y='spectroTemporalEntropyUndir',
#                           x_label='Days from deafening',
#                           y_label='Spectro temporal Entropy',
#                           title=f"Spectrotemporal Entropy (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='Spectro_temporal_entropy_across_days',
#                           xlim=[-40, 80], ylim=[0.2, 1],
#                           vline=0,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )
#
# # Entropy variance
# plot_across_days_per_note(df, x='taskSessionDeafening', y='entropyVarUndir',
#                           x_label='Days from deafening',
#                           y_label='Entropy variance',
#                           title=f"Entropy variance (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='EV_across_days',
#                           xlim=[-40, 80], ylim=[0, 0.04],
#                           vline=0,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )

# # Plot normalized values
# from analysis.functions import add_pre_normalized_col
#
# df_norm = add_pre_normalized_col(df, 'entropyUndir', 'entropyUndirNorm')
# df_norm = add_pre_normalized_col(df_norm, 'entropyDir', 'entropyDirNorm')
#
# df_norm = add_pre_normalized_col(df_norm, 'spectroTemporalEntropyUndir', 'spectroTemporalEntropyUndirNorm')
# df_norm = add_pre_normalized_col(df_norm, 'spectroTemporalEntropyDir', 'spectroTemporalEntropyDirNorm')
#
# df_norm = add_pre_normalized_col(df_norm, 'entropyVarUndir', 'entropyVarUndirNorm')
# df_norm = add_pre_normalized_col(df_norm, 'entropyVarDir', 'entropyVarDirNorm')
#
# plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='entropyUndirNorm',
#                           x_label='Days from deafening',
#                           y_label='Norm. Spectral Entropy',
#                           title=f"Norm. Entropy variance (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='Spectral_entropy_norm_across_days',
#                           x_lim=[0, 75], y_lim=[0.5, 1.5],
#                           hline=1,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )
#
# plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='spectroTemporalEntropyUndirNorm',
#                           x_label='Days from deafening',
#                           y_label='Norm. Spectro temporal Entropy',
#                           title=f"Norm. Spectro temporal Entropy (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='Spectral_entropy_norm_across_days',
#                           x_lim=[0, 75], y_lim=[0.5, 1.5],
#                           hline=1,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )
#
# plot_across_days_per_note(df_norm, x='taskSessionDeafening', y='entropyVarUndirNorm',
#                           x_label='Days from deafening',
#                           y_label='Norm. Entropy variance',
#                           title=f"Norm. Entropy variance (nb of notes >= {nb_note_crit}) Undir",
#                           fig_name='Entropy_variance_norm_across_days',
#                           x_lim=[0, 75], y_lim=[0, 2.5],
#                           hline=1,
#                           view_folder=True,
#                           save_fig=False,
#                           save_path=save_path
#                           )

# Compare conditional means of CV of FF
from deafening.plot import plot_paired_data

plot_paired_data(df_norm, x='taskName', y='entropyUndir',
                 x_label=None, y_label="Spectral Entropy",
                 y_lim=[0, 1.2],
                 view_folder=True,
                 fig_name='Spectral_entropy_comparison',
                 save_fig=False,
                 save_path=save_path,
                 fig_ext='.png'
                 )

plot_paired_data(df_norm, x='taskName', y='spectroTemporalEntropyUndir',
                 x_label=None, y_label='Spectro temporal Entropy',
                 y_lim=[0, 1.2],
                 view_folder=True,
                 fig_name='Spectro_temporal_entropy_across_days',
                 save_fig=False,
                 save_path=save_path,
                 fig_ext='.png'
                 )

plot_paired_data(df_norm, x='taskName', y='entropyVarUndir',
                 x_label=None, y_label="Entropy variance",
                 y_lim=[0, 0.03],
                 view_folder=True,
                 fig_name='EV_across_days',
                 save_fig=False,
                 save_path=save_path,
                 fig_ext='.png'
                 )

# df_mean.to_csv(save_path / 'df_mean.csv', index=False, header=True)


# Compare post-deafening values relative to pre-deafening baseline (1)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from util.draw import remove_right_top

df_mean = df.groupby(['birdID', 'note', 'taskName']).mean().reset_index()
df_mean = df_mean.query('taskName== "Postdeafening"')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0] = sns.stripplot(ax=axes[0], x=df_mean["taskName"], y=df_mean["entropyUndirNorm"],
                        color='k', jitter=0.05)
axes[0] = sns.boxplot(ax=axes[0], y=df_mean["entropyUndirNorm"],
                      width=0.2, color='w', showfliers = False)
axes[0].set_ylim([0, 1.5])
axes[0].set_title('Norm. Spectral Entropy')
axes[0].set_xlabel('')
axes[0].axhline(y=1, color='m', ls='--', lw=0.5)
remove_right_top(axes[0])

# One-sample t-test (one-tailed)
statistics = stats.ttest_1samp(a=df_mean["entropyUndirNorm"].dropna(), popmean=1, alternative='greater')
msg = f"t({len(df_mean['entropyUndirNorm'].dropna())-1})=" \
      f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
axes[0].text(-0.25, 0.1, msg, fontsize=12)


axes[1] = sns.stripplot(ax=axes[1], x=df_mean["taskName"], y=df_mean["spectroTemporalEntropyUndirNorm"],
                        color='k', jitter=0.05)
axes[1] = sns.boxplot(ax=axes[1], y=df_mean["spectroTemporalEntropyUndirNorm"],
                      width=0.2, color='w', showfliers = False)
axes[1].set_ylim([0, 1.5])
axes[1].set_title('Norm. Spectro-temporal Entropy')
axes[1].set_xlabel('')
axes[1].axhline(y=1, color='m', ls='--', lw=0.5)
remove_right_top(axes[1])

statistics = stats.ttest_1samp(a=df_mean["spectroTemporalEntropyUndirNorm"].dropna(), popmean=1, alternative='greater')
msg = f"t({len(df_mean['spectroTemporalEntropyUndirNorm'].dropna())-1})=" \
      f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
axes[1].text(-0.25, 0.1, msg, fontsize=12)

axes[2] = sns.stripplot(ax=axes[2], x=df_mean["taskName"], y=df_mean["entropyVarUndirNorm"],
                        color='k', jitter=0.05)
axes[2] = sns.boxplot(ax=axes[2], y=df_mean["entropyVarUndirNorm"],
                      width=0.2, color='w', showfliers = False)
axes[2].set_ylim([0, 1.5])
axes[2].set_title('Norm. Entropy Variance')
axes[2].set_xlabel('')
axes[2].axhline(y=1, color='m', ls='--', lw=0.5)
remove_right_top(axes[2])

statistics = stats.ttest_1samp(a=df_mean["entropyVarUndirNorm"].dropna(), popmean=1, alternative='less')
msg = f"t({len(df_mean['entropyVarUndirNorm'].dropna())-1})=" \
      f"{statistics.statistic: 0.3f}, p={statistics.pvalue: 0.3f}"
axes[2].text(-0.25, 0.1, msg, fontsize=12)

plt.show()
