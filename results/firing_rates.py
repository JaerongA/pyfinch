"""Compare motif firing rates between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from util import save
from results.plot import plot_bar_comparison


# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
df.set_index('id')

# Parameters
nb_row = 3
nb_col = 3
save_fig = True
fig_ext = '.png'

# Plot the results
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.text(0, 0.9, 'here', fontsize=15)
plt.suptitle('Firing Rates', y=.9, fontsize=20)

# Baseline FR
ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['baselineFR'], df['taskName'], hue_var=df['birdID'],
                    title='Baseline', ylabel='Firing Rates (Hz)',
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Undir
ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['motifFRUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Undir',
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Dir
ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['motifFRDir'], df['taskName'], hue_var=df['birdID'],
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