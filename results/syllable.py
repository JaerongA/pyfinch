"""Plot results from raster_syllable.py"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from math import ceil
from util import save
from util.draw import remove_right_top

# Load database
db = ProjectLoader().load_db()

def plot_fr_hist(save_fig=True, fig_ext='.png', bin_width=1,
                 hist_col='c',
                 edge_col='black'):

    # # SQL statement
    df = db.to_dataframe("SELECT * FROM syllable")
    df.set_index('syllableID')

    # Plot syllable FR histograms
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    fig.set_dpi(400)
    fig.suptitle('Syllable FR' +'\n', y=.995, fontsize=11)
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
        save.save_fig(fig, save_path, 'syllable_fr_hist', fig_ext=fig_ext)
    else:
        plt.show()


def plot_pcc_syllable_by_day(fr_criteria=0, save_fig=True):
    """
    Plot daily pcc per syllable per bird
    Parameters
    ----------
    fr_criteria : 0 by default
        only plot the pcc for syllable having higher firing rates than criteria
    Returns
    -------

    """
    import seaborn as sns

    # # SQL statement
    df = db.to_dataframe(f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria}")
    df.set_index('syllableID')
    # Plot the results

    circ_size = 1

    bird_list = sorted(set(df['birdID'].to_list()))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)

    fig.get_axes()[0].annotate(f"PCC syllable (Undir) FR >= {fr_criteria}", (0.5, 0.97),
                               xycoords='figure fraction',
                               ha='center',
                               fontsize=16)
    axes = axes.ravel()

    for bird, ax_ind in zip(bird_list, range(len(bird_list))):

        temp_df = df.loc[df['birdID'] == bird]
        # print(bird, ax_ind)
        # range =  [temp_df['taskSessionDeafening'].min(), temp_df['taskSessionDeafening'].max()]
        sns.lineplot(x='taskSessionDeafening', y='pccUndir', hue='note',
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

        axes[ax_ind].set_xlim([-17, 70])
        axes[ax_ind].set_ylim([-0.1, 0.6])

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, f'pcc_syllable_day(fr_over_{fr_criteria})', fig_ext=fig_ext)
    else:
        plt.show()


# Parameters
save_fig = True
fig_ext = '.png'
fr_criteria = 0

# plot_fr_hist(save_fig=save_fig)

# plot_pcc_syllable_by_day(fr_criteria=fr_criteria, save_fig=save_fig)

