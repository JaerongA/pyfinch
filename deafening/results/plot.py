# Functions used for plotting results

def get_nb_cluster(ax):
    from database.load import ProjectLoader
    from pandas.plotting import table

    # Load database
    db = ProjectLoader().load_db()
    # # SQL statement
    # Only the neurons that could be used in the analysis (bursting during undir & number of motifs >= 10)
    df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
    df.set_index('id')

    df_nb_cluster = df.groupby(['birdID', 'taskName']).count()['id'].reset_index()
    df_nb_cluster = df_nb_cluster.pivot_table('id', ['birdID'], 'taskName')
    df_nb_cluster = df_nb_cluster.fillna(0).astype(int)
    df_nb_cluster.loc['Total'] = df_nb_cluster.sum(numeric_only=True)
    df_nb_cluster = df_nb_cluster[['Predeafening', 'Postdeafening']]

    # Plot in a table format
    table(ax, df_nb_cluster,
          loc="center", colWidths=[0.2, 0.2, 0.2], cellLoc='center');


def plot_bar_comparison(ax, dependent_var, group_var, hue_var=None,
                        title=None, xlabel=None, ylabel=None,
                        col_order=None,
                        y_lim=None,
                        jitter=0.1, alpha=0.5,
                        run_stats=True, stat_txt_size=10,
                        legend_ok=False
                        ):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns

    dependent_var.replace('', np.nan,
                          inplace=True)  # replace empty cells with np.nans (to prevent the var to be recognized as non-numeric)

    if hue_var is not None:
        sns.stripplot(group_var, dependent_var, ax=ax,
                           size=5, hue=hue_var, jitter=jitter, order=col_order,
                           edgecolor="k", alpha=alpha, linewidth=1, zorder=1)
    else:
        sns.stripplot(group_var, dependent_var, ax=ax,
                           size=5, color="w", jitter=jitter, order=col_order,
                           edgecolor="k", alpha=alpha, linewidth=1, zorder=1)

    sns.barplot(group_var, dependent_var, ax=ax, facecolor=(1, 1, 1, 0),
                     linewidth=1,
                     order=col_order, errcolor=".2", edgecolor=".2", zorder=0)
    title += '\n\n\n'
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)

    # Add stat comparisons
    if run_stats:
        group1 = dependent_var[group_var == list(set(group_var))[0]].dropna()
        group2 = dependent_var[group_var == list(set(group_var))[1]].dropna()
        tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
        degree_of_freedom = len(group1) + len(group2) - 2

        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        x1, x2 = 0, 1
        y, h, col = ax.get_ylim()[1] * 1.02, 0, 'k'
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c=col)
        plt.text((x1 + x2) * .5, y + h * 1.1, sig, ha='center', va='bottom', color=col, size=stat_txt_size)
        if sig == '***':  # mark significance
            msg = ('$P$ < 0.001')
        else:
            msg = ('$P$ = {:.3f}'.format(pval))
        plt.text((x1 + x2) * .5, y * 1.1, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
        msg = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
        plt.text((x1 + x2) * .5, y * 1.2, msg, ha='center', va='bottom', color=col, size=stat_txt_size)

    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    # else:
    #     ax.set_ylim([0, myround(math.ceil(y), base=10)])
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    if legend_ok and hue_var is not None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.get_legend().remove()


def plot_cluster_pie_chart(axis, colors, category_column_name):
    pass


def plot_across_days_per_note(df, x, y,
                              x_label=None,
                              y_label=None,
                              title=None, fig_name=None,
                              xlim=None, ylim=None,
                              vline=None,
                              hline=None,
                              view_folder=True,
                              save_fig=True,
                              save_path=None,
                              fig_ext='.png'
                              ):
    # Load database
    import seaborn as sns
    import matplotlib.pyplot as plt
    from util.draw import remove_right_top
    from util import save

    # Plot the results
    circ_size = 1

    # bird_list = sorted(set(df['birdID'].to_list()))
    bird_list = df['birdID'].unique()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)

    fig.get_axes()[0].annotate(f"{title}", (0.5, 0.97),
                               xycoords='figure fraction',
                               ha='center',
                               fontsize=16)
    axes = axes.ravel()

    for bird, ax_ind in zip(bird_list, range(len(bird_list))):

        temp_df = df.loc[df['birdID'] == bird]
        sns.lineplot(x=x, y=y, hue='note',
                     data=temp_df, ci=None, marker='o', mew=circ_size, ax=axes[ax_ind])
        remove_right_top(axes[ax_ind])
        axes[ax_ind].set_title(bird)
        if ax_ind >= 5:
            axes[ax_ind].set_xlabel(x_label)
        else:
            axes[ax_ind].set_xlabel('')

        if (ax_ind == 0) or (ax_ind == 5):
            axes[ax_ind].set_ylabel(y_label)
        else:
            axes[ax_ind].set_ylabel('')

        if xlim:
            axes[ax_ind].set_xlim(xlim)
        if ylim:
            axes[ax_ind].set_ylim(ylim)

        if isinstance(vline, int):  # Plot vertical line
            axes[ax_ind].axvline(x=vline, color='k', ls='--', lw=0.5)
        if isinstance(hline, int):  # Plot horizontal line (e.g., baseline)
            axes[ax_ind].axhline(y=hline, color='k', ls='--', lw=0.5)

    if save_fig:
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()

def plot_paired_data(df, x, y,
                     x_label=None, y_label=None,
                     y_lim=None,
                     view_folder=True,
                     fig_name=None,
                     save_fig=True,
                     save_path=None,
                     fig_ext='.png'
                     ):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from util.draw import remove_right_top

    df_mean = df.groupby(['birdID', 'note', 'taskName']).mean()[y].reset_index()

    # Make paired-comparisons
    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    current_palette = sns.color_palette()
    inc = 0

    for bird in df_mean['birdID'].unique():
        for note in df_mean['note'].unique():
            temp_df = df_mean.loc[(df_mean['birdID'] == bird) & (df_mean['note'] == note)]
            if len(temp_df[x].unique()) == 2:  # paired value exists
                ax = sns.pointplot(x=x, y=y,
                                   data=temp_df,
                                   order=["Predeafening", "Postdeafening"],
                                   aspect=.5, scale=0.7, color=current_palette[inc])
            elif len(temp_df[x].unique()) == 1:  # paired value doesn't exist
                if temp_df[x].values == 'Predeafening':
                    ax.scatter(-0.2, temp_df[y].values, color=current_palette[inc])
                if temp_df[x].values == 'Postdeafening':
                    ax.scatter(1.2, temp_df[y].values, color=current_palette[inc])
        inc += 1
    #     ax.get_legend().remove()
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel(x_label), ax.set_ylabel(y_label)
    remove_right_top(ax)

    # 2 sample t-test
    group_var = df_mean[x]
    dependent_var = df_mean[y]
    group1 = dependent_var[group_var == list(set(group_var))[0]].dropna()
    group2 = dependent_var[group_var == list(set(group_var))[1]].dropna()
    tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
    degree_of_freedom = len(group1) + len(group2) - 2
    msg1 = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
    if pval < 0.001:  # mark significance
        msg2 = ('p < 0.001')
    else:
        msg2 = ('p = {:.3f}'.format(pval))
    msg = msg1 + ', ' + msg2

    # rank-sum
    # group1, group2 = [], []
    # group1 = df_mean.query('taskName == "Predeafening"')['entropyUndir']
    # group2 = df_mean.query('taskName == "Postdeafening"')['entropyUndir']
    # stat, pval = stats.ranksums(group1, group2)
    # degree_of_freedom = len(group1) + len(group2) - 2
    # msg = f"ranksum p-val = {pval : .3f}"

    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    x1, x2 = 0, 1
    y_loc, h, col = df_mean[y].max() + 0.2, 0.05, 'k'

    ax.plot([x1, x1, x2, x2], [y_loc, y_loc + h, y_loc + h, y_loc], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y_loc + h * 1.5, sig, ha='center', va='bottom', color=col, size=10)
    plt.title(msg, size=10)

    if save_fig:
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()

# from database.load import ProjectLoader
# import matplotlib.pyplot as plt
# from util import save
#
# # Load database
# db = ProjectLoader().load_db()
# # # SQL statement
# df = db.to_dataframe("SELECT unitCategoryUndir FROM cluster WHERE ephysOK=TRUE")
# unit_category = df['unitCategoryUndir']
#
# explode = (0.1, 0)
# colors = ['#66b3ff', '#ff9999']
# values = [sum(unit_category == 'Bursting'), sum(unit_category == 'NonBursting')]
#
# fig, ax = plt.subplots()
# ax.pie(values, explode=explode, colors=colors,
#         shadow=True, labels=unit_category.unique(), startangle=90,
#         autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))
#
# plt.title('Unit Category (Undir)')
# ax.axis('equal')
# plt.show()
