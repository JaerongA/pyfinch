"""Plot results from raster_syllable.py"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from math import ceil
from util import save
from util.draw import remove_right_top

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM syllable")
df.set_index('syllableID')


def plot_fr_hist(save_fig=True, fig_ext='.png', bin_width=1,
                 hist_col='c',
                 edge_col='black'):
    # Plot syllable FR histograms
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    fig.set_dpi(400)
    fig.suptitle('Syllable FR', y=.995, fontsize=11)
    ax[0].set_title('Undir')
    ax[0].hist(df['frUndir'], bins=range(0, ceil(max(df['frUndir'])) + bin_width, bin_width),
               color=hist_col, edgecolor=edge_col)
    ax[0].set_xlabel('FR'), ax[0].set_ylabel('Count')
    remove_right_top(ax[0])

    ax[1].set_title('Dir')
    ax[1].hist(df['frDir'], bins=range(0, ceil(max(df['frUndir'])) + bin_width, bin_width),
               color=hist_col, edgecolor=edge_col)
    ax[1].set_xlabel('FR')
    remove_right_top(ax[1])
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, 'SyllableFR', fig_ext=fig_ext)
    else:
        plt.show()


plot_fr_hist()

# Plot the results
# fig, ax = plt.subplots(figsize=(10, 4))
# plt.suptitle('Syllable PCC', y=.9, fontsize=20)


# # Save results
# if save_fig:
#     save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
#     save.save_fig(fig, save_path, 'PCC', fig_ext=fig_ext)
# else:
#     plt.show()
