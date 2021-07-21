"""Compare pairwise cross-correlation (pcc) between different conditions"""

import matplotlib.pyplot as plt
from results.plot import plot_bar_comparison
import seaborn as sns
from util import save
import numpy as np

from database.load import ProjectLoader
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from util.draw import remove_right_top

# Parameters
nb_row = 3
nb_col = 4
save_fig = False
fig_ext = '.png'
fr_criteria = 10



def plot_pcc_regression(x, y,
                        x_label, y_label,
                        fr_criteria=fr_criteria, save_fig=save_fig, regression_fit=True):

    fig, ax = plt.subplots(figsize=(5, 4))

    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    ax.scatter(x, y, color='k')
    ax.set_title(f"Undir FR over {fr_criteria}")
    ax.set_xlabel('dph before deafening')
    ax.set_xlabel('Days from deafening')
    ax.set_ylabel('Syllable PCC')
    ax.set_ylim([-0.1, 0.5])

    if regression_fit:
        # Regression analysis
        model = LinearRegression().fit(x, y).predict(x)
        ax.plot(x, model, color='r')
        x = df['dph']
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

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, f'pcc_syllable_reg(fr_over_{fr_criteria})', fig_ext=fig_ext)
    else:
        plt.show()


# # Load database
# db = ProjectLoader().load_db()
# # # SQL statement
# df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
# df.set_index('id')
#
# # Plot the results
# fig, ax = plt.subplots(figsize=(10, 4))
# plt.suptitle('Pairwise CC', y=.9, fontsize=20)
#
# # Undir
# df['pairwiseCorrUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
# ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['pairwiseCorrUndir'], df['taskName'], hue_var=df['birdID'],
#                     title='Undir', ylabel='PCC',
#                     y_max=round(df['pairwiseCorrUndir'].max() * 10) / 10 + 0.1,
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
#
# # Dir
# df['pairwiseCorrDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
# ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
# plot_bar_comparison(ax, df['pairwiseCorrDir'], df['taskName'], hue_var=df['birdID'],
#                     title='Dir', y_max=round(df['pairwiseCorrDir'].max() * 10) / 10 + 0.2,
#                     col_order=("Predeafening", "Postdeafening"),
#                     )
# fig.tight_layout()
#
# # Undir (paired comparisons)
# pcc_mean_per_condition = df.groupby(['birdID', 'taskName'])['pairwiseCorrUndir'].mean().to_frame()
# pcc_mean_per_condition.reset_index(inplace=True)
#
# ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)
#
# ax = sns.pointplot(x='taskName', y='pairwiseCorrUndir', hue='birdID',
#                    data=pcc_mean_per_condition,
#                    order=["Predeafening", "Postdeafening"],
#                    aspect=.5, hue_order=df['birdID'].unique().tolist(), scale=0.7)
#
# ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
#
# title = 'Undir (Paired Comparison)'
# title += '\n\n\n'
#
# plt.title(title)
# plt.xlabel(''), plt.ylabel('')
# plt.ylim(0, 0.3), plt.xlim(-0.5, 1.5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# # Save results
# if save_fig:
#     save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
#     save.save_fig(fig, save_path, 'PCC', fig_ext=fig_ext)
# else:
#     plt.show()

## Syllable PCC plot across days (with regression)
## SQL statement
# query = f"SELECT * FROM syllable WHERE frUndir >= {fr_criteria} AND taskSessionDeafening > 0 AND taskSessionDeafening < 40"
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskSessionDeafening <= 0"

# Load database
db = ProjectLoader().load_db()
df = db.to_dataframe(query)
df.set_index('syllableID')
# x = df['taskSessionDeafening'].values.reshape(-1, 1)
x = df['dph'].values.reshape(-1, 1)
y = df['pccUndir'].values.reshape(-1, 1)


plot_pcc_regression('dph',
    fr_criteria=fr_criteria, save_fig=save_fig, regression_fit=True)
