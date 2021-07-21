"""Plot results from raster_syllable.py"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from math import ceil
from util import save
from util.draw import remove_right_top

# Parameters
save_fig = True
fig_ext = '.png'
fr_criteria = 10

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
    fig.suptitle('Syllable FR' + '\n', y=.995, fontsize=11)
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


# plot_fr_hist(save_fig=save_fig)

plot_pcc_syllable_by_day(fr_criteria=fr_criteria, save_fig=save_fig)



# Parameters
# nb_row = 3
# nb_col = 2
#
# from results.plot import plot_bar_comparison
# import numpy as np
# import seaborn as sns
#
# # # SQL statement
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria}"
# # query = f"SELECT * FROM syllable"
#
# df = db.to_dataframe(query)
# # Plot the results
# fig, ax = plt.subplots(figsize=(6, 4))
# plt.suptitle('Pairwise CC', y=.9, fontsize=20)
#
# # Undir
# df['pccUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
# ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['pccUndir'], df['taskName'], hue_var=df['birdID'],
#                     title='Undir', ylabel='PCC',
#                     y_max=round(df['pccUndir'].max() * 10) / 10 + 0.1,
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Dir
# df['pccDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
# ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['pccDir'], df['taskName'], hue_var=df['birdID'],
#                     title='Dir', y_max=round(df['pccDir'].max() * 10) / 10 + 0.2,
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
# fig.tight_layout()
#
# plt.show()

#
# # Undir (paired comparisons)
# pcc_mean_per_condition = df.groupby(['birdID','taskName'])['pccUndir'].mean().to_frame()
# pcc_mean_per_condition.reset_index(inplace = True)
#
# ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
#
# ax = sns.pointplot(x='taskName', y='pccUndir', hue = 'birdID',
#                    data=pcc_mean_per_condition,
#                    order=["Predeafening", "Postdeafening"],
#                    aspect=.5, hue_order = df['birdID'].unique().tolist(), scale = 0.7)
#
# ax.spines['right'].set_visible(False),ax.spines['top'].set_visible(False)
#
# title = 'Undir (Paired Comparison)'
# title += '\n\n\n'
# plt.title(title)
# plt.xlabel(''), plt.ylabel('')
# plt.ylim(0, 0.3), plt.xlim(-0.5, 1.5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



# from results.plot import plot_bar_comparison
# import numpy as np
# import seaborn as sns
#
# # # SQL statement
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria}"
# # query = f"SELECT * FROM syllable"
#
# mode = 'violin'
# # mode = 'bar'
#
# df = db.to_dataframe(query)
# # Plot the results
#
# # Undir
# x = df['block10days']
# y = df['pccUndir']
# # dependent_var = df['pccDir']
#
# x.replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
#
# fig, ax = plt.subplots(figsize=(5, 4))
#
# if mode == 'bar':
#     ax = sns.barplot(x, y, ax=ax, facecolor=(1, 1, 1, 0),
#                      linewidth=1,
#                      errcolor=".2", edgecolor=".2", zorder=0)
# elif mode == 'violin':
#     ax = sns.violinplot(x, y, inner=None)
#     ax = sns.swarmplot(x, y, color="k")
#
# remove_right_top(ax)
#
# title = 'Syllable PCC per block (Undir)'
# title += '\n\n'
# plt.title(title), plt.xlabel('Day Block (10 days)'), plt.ylabel('PCC')
# # ax.set_ylim([0, 0.2])
# day_block_label_list = ['Predeafening', 'Day 1-10', 'Day 11-20', 'Day 21-30', 'Day >= 31' ]
# ax.set_xticklabels(day_block_label_list)
# plt.xticks(rotation=45)
# plt.show()


regression_fit = True

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# # SQL statement
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening > 0 AND taskSessionDeafening < 40"
query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening > 0"
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening < 40"
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria}"

df = db.to_dataframe(query)
df.set_index('syllableID')

fig, ax = plt.subplots(figsize=(5, 4))

# x = df['taskSessionDeafening'].values.reshape(-1, 1)
x = df['entropyUndir'].values.reshape(-1, 1)
y = df['pccUndir'].values.reshape(-1, 1)
ax.scatter(x, y, color='k')
ax.set_title(f"Undir FR over {fr_criteria}")
# ax.set_xlabel('dph before deafening')
# ax.set_xlabel('Days from deafening')
ax.set_xlabel('Entropy')
ax.set_ylabel('Syllable PCC')
# ax.set_ylim([-0.1, 0.5])

if regression_fit:
    # Regression analysis
    model = LinearRegression().fit(x, y).predict(x)
    ax.plot(x, model, color='r')
    # x = df['dph']
    x = df['entropyUndir']
    # x = df['taskSessionDeafening']
    y = df['pccUndir']
    corr, corr_pval = pearsonr(x, y)

    txt_xloc = 0.7
    txt_yloc = 0.85
    txt_inc = 0.05
    fig.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 3)}", fontsize=10)
    txt_yloc -= txt_inc
    t = fig.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 3)}", fontsize=10)

remove_right_top(ax)

plt.show()
