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
