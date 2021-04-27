def get_nb_cluster():
    from database.load import ProjectLoader
    from pandas.plotting import table
    import matplotlib.pyplot as plt
    from util import save

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

    fig, ax = plt.subplots(1, 1)
    plt.title("# of clusters")
    table(ax, df_nb_cluster, loc="center", colWidths=[0.2, 0.2, 0.2]);
    plt.axis('off')
    plt.show()


def plot_cluster_pie_chart(axis, colors, category_column_name):
    pass


# def plot_bar_comparison():
# #     # plot a bar chart with a stat test
# #     pass


# get_nb_cluster()

# category_column_name = 'unitCategoryUndir'
#
# from database.load import ProjectLoader
# import matplotlib.pyplot as plt
#
# # Load database
# db = ProjectLoader().load_db()
# # # SQL statement
# df = db.to_dataframe("SELECT * FROM cluster WHERE ephysOK=TRUE")
# df.set_index('id')
# unit_category = df[category_column_name].dropna()
#
# # Plot pie chart
# explode = (0.1, 0)
# colors = ['#66b3ff', '#ff9999']
# values = [sum(unit_category == 'Bursting'), sum(unit_category == 'NonBursting')]
# fig, ax = plt.subplots()
# ax.pie(values, explode=explode, colors=colors,
#         shadow=True, labels=unit_category.unique(), startangle=90,
#         autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))
#
# plt.title('Cluster Category')
# ax.axis('equal')
#
# plt.show()
#
# # summary = summary[summary['UnitCategory_Undir'] == 'Bursting']
# # labels=UnitCategory.unique()
# # print(labels)


from database.load import ProjectLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Baseline


def plot_bar_comparison(ax, dependent_var, group_var, hue_var=None,
                        title=None, xlabel=None, ylabel=None,
                        col_order=None,
                        jitter=0.1, alpha=0.5,
                        save_fig=False
                        ):

    from util import save

    ax = sns.stripplot(group_var, dependent_var,
                       size=5, hue=hue_var, jitter=jitter, order=col_order,
                       edgecolor="gray", alpha=alpha, linewidth=1)

    ax = sns.barplot(group_var, dependent_var, ax=ax, facecolor=(1, 1, 1, 0),
                     linewidth=bar_line_width,
                     order=col_order, errcolor=".2", edgecolor=".2")

    plt.title(title)
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.ylim(0, y_max)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.get_legend().remove()

    # # Stats
    # pre_deafening = df.query('taskName == "Predeafening"')['baselineFR'].dropna()
    # post_deafening = df.query('taskName == "Postdeafening"')['baselineFR'].dropna()
    # tval, pval = stats.ttest_ind(pre_deafening, post_deafening)
    # degree_of_freedom = len(pre_deafening) + len(post_deafening) - 2
    #
    # if pval < 0.001:
    #     sig = '***'
    # elif pval < 0.01:
    #     sig = '**'
    # elif pval < 0.05:
    #     sig = '*'
    # else:
    #     sig = 'ns'
    #
    # x1, x2 = 0, 1
    # y, h, col = df['motifFRUndir'].max() + 3, 1, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text((x1 + x2) * .5, y + h * 1, sig, ha='center', va='bottom', color=col, size=15)
    # msg = ('$P$ = {:.4f}'.format(pval))
    # plt.text((x1 + x2) * .5, y_max * 1.2, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
    # msg = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
    # plt.text((x1 + x2) * .5, y_max * 1.3, msg, ha='center', va='bottom', color=col, size=stat_txt_size)

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'UnitProfiling')
        save.save_fig(fig, save_path, mi.name, fig_ext=fig_ext)
    else:
        plt.show()

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
df.set_index('id')

# Parameters
nb_row = 3
nb_col = 3
bar_line_width = 2
y_max = 60
stat_txt_size = 12

plt.figure(figsize=(11, 4))
plt.subplot(1, 3, 1)
plt.suptitle('Firing Rates', y=.95)
ax = plt.subplot2grid((nb_row, nb_col), (0, 0), rowspan=2, colspan=1)

plot_bar_comparison(ax, df['baselineFR'], df['taskName'], hue_var=df['birdID'],
                    title='Baseline', ylabel='Firing Rates (Hz)',
                    col_order=("Predeafening", "Postdeafening"),
                    jitter=0.1,
                    )


# # Undir
# plt.subplot(1, 3, 2)
# ax = sns.stripplot(df.TaskName, df.MotifFR_Undir,
#                    size=5, hue=df.BirdID, jitter=True, order=["Predeafening", "Postdeafening"],
#                    edgecolor="gray", alpha=.5, linewidth=1)
#
# ax = sns.barplot(df.TaskName, df.MotifFR_Undir, facecolor=(1, 1, 1, 0),
#                  linewidth=bar_linewidth,
#                  order=["Predeafening", "Postdeafening"], errcolor=".2", edgecolor=".2")
#
# plt.title('Undir')
# plt.xlabel(''), plt.ylabel('')
# plt.ylim(-1, y_max)
# ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# pre_deafening, post_deafening = [], []
# pre_deafening = df.query('TaskName == "Predeafening"')['MotifFR_Undir'].dropna()
# post_deafening = df.query('TaskName == "Postdeafening"')['MotifFR_Undir'].dropna()
# tval, pval = stats.ttest_ind(pre_deafening, post_deafening)
# df = len(pre_deafening) + len(post_deafening) - 2
#
# # msg = ('t = {:.3f}'.format(tval)' $P$ = {:.3f}'.format(pval))
#
#
# # 't = ({:.3f}'.format(tval),
# if pval < 0.001:
#     sig = '***'
# elif pval < 0.01:
#     sig = '**'
# elif pval < 0.05:
#     sig = '*'
# else:
#     sig = 'ns'
#
# x1, x2 = 0, 1
# y, h, col = df['MotifFR_Undir'].max() + 3, 1, 'k'
# plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
# plt.text((x1 + x2) * .5, y + h * 2, sig, ha='center', va='bottom', color=col, size=12)
# msg = ('$P$ = {:.4f}'.format(pval))
# plt.text((x1 + x2) * .5, y_max * 1.2, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
# msg = ('t({:.0f})'.format(df) + ' = {:.2f}'.format(tval))
# plt.text((x1 + x2) * .5, y_max * 1.3, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
#
# # plt.tight_layout()
