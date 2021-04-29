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
                        x_max=0, y_max=None,
                        jitter=0.1, alpha=0.5,
                        run_stats=True, stat_txt_size = 10,
                        legend_ok=False
                        ):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    from util.functions import myround

    dependent_var.replace('', np.nan,
                          inplace=True)  # replace empty cells with np.nans (to prevent the var to be recognized as non-numeric)

    if hue_var is not None:
        ax = sns.stripplot(group_var, dependent_var,
                           size=5, hue=hue_var, jitter=jitter, order=col_order,
                           edgecolor="k", alpha=alpha, linewidth=1, zorder=1)
    else:
        ax = sns.stripplot(group_var, dependent_var,
                           size=5, color="w", jitter=jitter, order=col_order,
                           edgecolor="k", alpha=alpha, linewidth=1, zorder=1)

    ax = sns.barplot(group_var, dependent_var, ax=ax, facecolor=(1, 1, 1, 0),
                     linewidth=1,
                     order=col_order, errcolor=".2", edgecolor=".2", zorder=0)
    title += '\n\n\n'
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)

    # Add stat comparisons
    if run_stats:
        group1 = dependent_var[group_var == list(set(group_var))[0]]
        group2 = dependent_var[group_var == list(set(group_var))[1]]
        tval, pval = stats.ttest_ind(group1, group2, nan_policy='omit')
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
        y, h, col = ax.get_ylim()[1] * 1.01, 0.2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c=col)
        plt.text((x1 + x2) * .5, y + h * 1, sig, ha='center', va='bottom', color=col, size=stat_txt_size)
        msg = ('$P$ = {:.3f}'.format(pval))
        plt.text((x1 + x2) * .5, y * 1.1, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
        msg = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
        plt.text((x1 + x2) * .5, y * 1.2, msg, ha='center', va='bottom', color=col, size=stat_txt_size)

    if y_max:
        plt.ylim(x_max, y_max)
    else:
        ax.set_ylim([0, myround(math.ceil(y), base=10)])
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    if legend_ok and hue_var is not None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.get_legend().remove()


def plot_cluster_pie_chart(axis, colors, category_column_name):
    pass




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
