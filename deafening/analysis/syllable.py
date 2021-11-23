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




# Parameters
save_fig = False
fig_ext = '.png'
fr_criteria = 10


plot_fr_hist(save_fig=save_fig)

#plot_pcc_syllable_by_day(fr_criteria=fr_criteria, save_fig=save_fig)






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


# regression_fit = True
#
# from sklearn.linear_model import LinearRegression
# from scipy.stats import pearsonr
#
# # # SQL statement
# # query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening > 0 AND taskSessionDeafening < 40"
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening > 0"
# # query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening < 40"
# # query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria}"
#
# df = db.to_dataframe(query)
# df.set_index('syllableID')
#
# fig, ax = plt.subplots(figsize=(5, 4))
#
# # x = df['taskSessionDeafening'].values.reshape(-1, 1)
# x = df['entropyUndir'].values.reshape(-1, 1)
# y = df['pccUndir'].values.reshape(-1, 1)
# ax.scatter(x, y, color='k')
# ax.set_title(f"Undir FR over {fr_criteria}")
# # ax.set_xlabel('dph before deafening')
# # ax.set_xlabel('Days from deafening')
# ax.set_xlabel('Entropy')
# ax.set_ylabel('Syllable PCC')
# # ax.set_ylim([-0.1, 0.5])
#
# if regression_fit:
#     # Regression analysis
#     model = LinearRegression().fit(x, y).predict(x)
#     ax.plot(x, model, color='r')
#     # x = df['dph']
#     x = df['entropyUndir']
#     # x = df['taskSessionDeafening']
#     y = df['pccUndir']
#     corr, corr_pval = pearsonr(x, y)
#
#     txt_xloc = 0.7
#     txt_yloc = 0.85
#     txt_inc = 0.05
#     fig.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 3)}", fontsize=10)
#     txt_yloc -= txt_inc
#     t = fig.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 3)}", fontsize=10)
#
# remove_right_top(ax)
#
# plt.show()


