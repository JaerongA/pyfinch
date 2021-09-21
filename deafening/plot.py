"""Plotting functions & classes specific to the deafening project"""


def plot_paired_scatter(df, x, y,
                        save_folder_name,
                        x_lim=None,
                        y_lim=None,
                        x_label=None,
                        y_label=None,
                        title=None,
                        diagonal=True,  # plot diagonal line
                        paired_test=True,
                        save_fig=True,
                        view_folder=False,
                        fig_ext='.png'):

    from database.load import ProjectLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import ttest_rel
    from util.draw import remove_right_top
    from util import save

    # Parameters
    nb_row = 3
    nb_col = 2

    # Plot scatter with diagonal
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(title, y=.9, fontsize=15)
    task_list = ['Predeafening', 'Postdeafening']

    for ind, task in enumerate(task_list):

        df_temp = df[df['taskName'] == task]
        ax = plt.subplot2grid((nb_row, nb_col), (1, ind), rowspan=2, colspan=1)
        sns.scatterplot(ax=ax, x=x, y=y, data=df_temp, size=2, color='k')
        if diagonal:
            ax.plot([0, 1], [0, 1], 'm--', transform=ax.transAxes, linewidth=1)

        if paired_test:
            stat = ttest_rel(df_temp[x], df_temp[y])
            msg = f"t({len(df_temp[x].dropna()) - 2})=" \
                  f"{stat.statistic: 0.3f}, p={stat.pvalue: 0.3f}"
            ax.text(1, 0.5, msg, fontsize=10)

        remove_right_top(ax)
        ax.set_aspect('equal')
        ax.set_xlabel(x_label), ax.set_ylabel(y_label), ax.set_title(task)
        ax.set_xlim(x_lim), ax.set_ylim(y_lim)
        ax.get_legend().remove()
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1, 1))
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + 1, 1))

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, save_folder_name, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()
