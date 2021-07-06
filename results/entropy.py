"""Compare pairwise cross-correlation (pcc) between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from results.plot import plot_bar_comparison
import seaborn as sns
from util import save
import numpy as np
from util.draw import remove_right_top

# Parameters
nb_row = 3
nb_col = 4
save_fig = False
fig_ext = '.png'
nb_note_crit = 10

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe(
    f"SELECT syl.*, song.taskSession, song.taskSessionDeafening, song.taskSessionPostDeafening, song.dph, song.block10days "
    f"FROM syllable syl INNER JOIN song ON syl.songID = song.id WHERE syl.nbNoteUndir >= {nb_note_crit}")
df.set_index('syllableID')

# Plot the results
circ_size = 0.5
# bird_list = sorted(set(df['birdID'].to_list()))
bird_list = df['birdID'].unique()
fig, axes = plt.subplots(2, 5, figsize=(21, 8))
fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)

fig.get_axes()[0].annotate(f"Spectro-temporal Entropy (nb of notes >= {nb_note_crit}) Undir", (0.5, 0.97),
                           xycoords='figure fraction',
                           ha='center',
                           fontsize=16)
axes = axes.ravel()

for bird, ax_ind in zip(bird_list, range(len(bird_list))):

    temp_df = df.loc[df['birdID'] == bird]
    sns.lineplot(x='taskSessionDeafening', y='spectroTemporalEntropyUndir', hue='note',
                 data=temp_df, ci=None, marker='o', mew=circ_size, ax=axes[ax_ind])
    remove_right_top(axes[ax_ind])
    axes[ax_ind].set_title(bird)
    if ax_ind >= 5:
        axes[ax_ind].set_xlabel('Days from deafening')
    else:
        axes[ax_ind].set_xlabel('')

    if (ax_ind == 0) or (ax_ind == 5):
        axes[ax_ind].set_ylabel('PCC')
    else:
        axes[ax_ind].set_ylabel('')

    axes[ax_ind].set_xlim([-30, 70])
    axes[ax_ind].set_ylim([0.25, 0.9])
    axes[ax_ind].axvline(x=0, color='k', linestyle='dashed', linewidth=0.5)

# Save figure
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'Entropy', fig_ext=fig_ext, view_folder=False)
else:
    plt.show()


