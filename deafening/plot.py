"""Plotting functions & classes specific to the deafening project"""
import matplotlib.pyplot as plt
import seaborn as sns
from util.draw import remove_right_top
from util import save


def plot_paired_scatter(df, x, y, hue=None,
                        save_folder_name=None,
                        x_lim=None, y_lim=None,
                        x_label=None, y_label=None, tick_freq=0.1,
                        title=None,
                        diagonal=True,  # plot diagonal line
                        paired_test=True,
                        save_fig=True,
                        view_folder=False,
                        fig_ext='.png'):
    from database.load import ProjectLoader
    import numpy as np
    from scipy.stats import ttest_rel

    # Parameters
    nb_row = 5
    nb_col = 2

    # Plot scatter with diagonal
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.suptitle(title, y=.9, fontsize=15)
    task_list = ['Predeafening', 'Postdeafening']
    ax_list = []
    for ind, task in enumerate(task_list):

        df_temp = df[df['taskName'] == task]
        ax = plt.subplot2grid((nb_row, nb_col), (1, ind), rowspan=3, colspan=1)

        if hue:  # color-code birds
            if ind == 0:
                ax0 = sns.scatterplot(ax=ax, x=x, y=y, hue=hue, data=df_temp, size=2, color='k')
            else:
                ax1 = sns.scatterplot(ax=ax, x=x, y=y, hue=hue, data=df_temp, size=2, color='k')
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            sns.scatterplot(ax=ax, x=x, y=y, data=df_temp, size=2, color='k')
            ax.get_legend().remove()

        if diagonal:
            ax.plot([0, 1], [0, 1], 'm--', transform=ax.transAxes, linewidth=1)

        if paired_test:
            ax_txt = plt.subplot2grid((nb_row, nb_col), (4, ind), rowspan=1, colspan=1)
            stat = ttest_rel(df_temp[x], df_temp[y])
            msg = f"t({len(df_temp[x].dropna()) - 2})=" \
                  f"{stat.statistic: 0.3f}, p={stat.pvalue: 0.3f}"
            ax_txt.text(0.25, 0.5, msg), ax_txt.axis('off')

        remove_right_top(ax)
        ax.set_aspect('equal')
        ax.set_xlabel(x_label), ax.set_ylabel(y_label), ax.set_title(task)
        ax.set_xlim(x_lim), ax.set_ylim(y_lim)
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + tick_freq, tick_freq))
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + tick_freq, tick_freq))

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, save_folder_name, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()


def plot_by_day_per_syllable(fr_criteria=0, save_fig=True):
    """
    Plot daily pcc per syllable per bird
    Parameters
    ----------
    fr_criteria : 0 by default
        only plot the pcc for syllable having higher firing rates than criteria
    Returns
    -------

    """

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
        # sns.lineplot(x='taskSessionDeafening', y='pccUndir', hue='note',
        sns.lineplot(x='taskSessionDeafening', y='entropyUndir', hue='note',
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
        # axes[ax_ind].set_ylim([-0.1, 0.6])

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, f'pcc_syllable_day(fr_over_{fr_criteria})', fig_ext=fig_ext)
    else:
        plt.show()
