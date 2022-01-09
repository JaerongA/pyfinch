"""
By Jaerong
Analyze unit profiles such as burst fraction
"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.plot import plot_bar_comparison, plot_per_day_block
from util import save

# Parameters
nb_row = 3
nb_col = 5
save_fig = False
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe(
    f"SELECT unit.*, cluster.taskSession, cluster.taskSessionDeafening, cluster.taskSessionPostDeafening, cluster.dph, cluster.block10days "
    f"FROM unit_profile unit INNER JOIN cluster ON cluster.id = unit.clusterID WHERE cluster.analysisOK=TRUE")
df.set_index('clusterID')

# Plot the results
fig, ax = plt.subplots(figsize=(14, 4))
plt.suptitle('Bursting Analysis', y=.95, fontsize=15)

# Burst Fraction
# df['pairwiseCorrUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['burstFractionUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Burst Fraction', y_label='Burst Fraction (%)',
                    y_lim=[0, 80],
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Burst Duration
ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['burstDurationUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Burst Duration', y_label='Burst Duration (ms)',
                    y_lim=[0, 10],
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Burst Freq
ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['burstFreqUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Burst Freq', y_label='Burst Freq (Hz)',
                    y_lim=[0, 12],
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Nb of spk per burst
ax = plt.subplot2grid((nb_row, nb_col), (1, 3), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['burstMeanNbSpkUndir'], df['taskName'], hue_var=df['birdID'],
                    title='# of spk per burst', y_label='# of spk',
                    y_lim=[0, 5],
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Burst index
ax = plt.subplot2grid((nb_row, nb_col), (1, 4), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['burstIndexUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Burst index', y_label='Burst index',
                    y_lim=[0, 0.3],
                    col_order=("Predeafening", "Postdeafening"),
                    )

fig.tight_layout()

# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'UnitProfile', fig_ext=fig_ext)
else:
    plt.show()

# Plot burst fraction per day blocks
plot_per_day_block(df, ind_var_name='block10days', dep_var_name='burstFractionUndir',
                   title='Burst Fraction (Undir) per day block',
                   y_label='Burst Fraction (%)', y_lim=[0, 100],
                   fig_name='BurstFraction_Undir_per_day_block',
                   save_fig=False, fig_ext='.png'
                   )
