"""Compare pairwise cross-correlation (pcc) between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from results.plot import plot_bar_comparison
import seaborn as sns
from util import save
import numpy as np

# Parameters
nb_row = 3
nb_col = 4
save_fig = False
fig_ext = '.png'
fr_criteria = 10


# def plot_pcc_regression(x, y,
#                         x_label, y_label, title,
#                         x_lim=None, y_lim=None,
#                         fr_criteria=fr_criteria, save_fig=save_fig, regression_fit=True):

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

def plot_regression(x, y, **kwargs):
    """Plot scatter plot between two continuous variables with its regression fit """

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.stats import pearsonr
    from util.draw import remove_right_top
    import statsmodels.api as sm

    def get_aic(x, y):
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        regr = OLS(y, add_constant(x)).fit()
        return regr.aic

    fig, ax = plt.subplots(figsize=(5, 4))

    x = x.values.T
    y = y.values.T

    x_ind = x.argsort()
    x = x[x_ind]
    y = y[x_ind]

    ax.scatter(x, y, color='k')
    remove_right_top(ax)

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'x_label' in kwargs:
        ax.set_xlabel(kwargs['x_label'])
    if 'y_label' in kwargs:
        ax.set_ylabel(kwargs['y_label'])

    txt_xloc = 0.6
    txt_yloc = 0.85
    txt_inc = 0.05

    # Regression analysis
    if 'regression_fit' in kwargs:
        for fit in kwargs['regression_fit']:

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            if fit == 'linear':
                # Linear regression

                corr, corr_pval = pearsonr(x.T[0], y.T[0])
                y_pred = LinearRegression().fit(x, y).predict(x)
                ax.plot(x, y_pred, color='r')
                aic_lin = get_aic(x, y_pred)
                fig.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 4)}", fontsize=10)
                txt_yloc -= txt_inc
                fig.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 4)}", fontsize=10)
                txt_yloc -= txt_inc
                fig.text(txt_xloc, txt_yloc, f"aic (linear) = {round(aic_lin, 3)}", fontsize=10)

            if fit == 'quadratic':
                # Polynomial regression
                poly = PolynomialFeatures(degree=2)
                x_transformed = poly.fit_transform(x)
                model = sm.OLS(y, x_transformed).fit()
                y_pred = model.predict(x_transformed)
                aic_quad = get_aic(x, y_pred)
                ax.plot(x, y_pred, color='cyan')
                txt_yloc -= txt_inc
                fig.text(txt_xloc, txt_yloc, f"aic (quad) = {round(aic_quad, 3)}", fontsize=10)

    if 'x_lim' in kwargs:
        ax.set_xlim(kwargs['x_lim'])
    if 'y_lim' in kwargs:
        ax.set_ylim(kwargs['y_lim'])

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, f'pcc_syllable_reg(fr_over_{fr_criteria})', fig_ext=fig_ext)
    else:
        plt.show()


# Load database
db = ProjectLoader().load_db()
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria}"
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskSessionDeafening <= 0"
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskName='Postdeafening'"
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_criteria} AND taskName='Postdeafening' AND taskSessionDeafening <= 40"
df = db.to_dataframe(query)
df.set_index('syllableID')

# # DPH
# x = df['dph']
# y = df['pccUndir']
#
# title = f'Undir FR over {fr_criteria} Postdeafening'
# x_label = 'Age (dph)'
# y_label = 'PCC'
# # x_lim = [90, 130]
# y_lim = [-0.05, 0.3]
#
# plot_regression(x, y,
#                     title=title,
#                     x_label=x_label, y_label=y_label,
#                     # x_lim=x_lim,
#                     y_lim=y_lim,
#                     fr_criteria=fr_criteria,
#                     save_fig=save_fig,
#                     regression_fit=True
#                     )


# Days from deafening
x = df['taskSessionDeafening']
y = df['pccUndir']

title = f'Undir FR over {fr_criteria}'
x_label = 'Days from deafening'
y_label = 'PCC'
# x_lim = [0, 35]
y_lim = [-0.05, 0.25]

plot_regression(x, y,
                title=title,
                x_label=x_label, y_label=y_label,
                # x_lim=x_lim,
                y_lim=y_lim,
                fr_criteria=fr_criteria,
                save_fig=save_fig,
                regression_fit={'linear', 'quadratic'}
                )
