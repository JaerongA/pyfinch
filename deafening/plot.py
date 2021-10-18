"""Plotting functions & classes specific to the deafening project"""

from database.load import ProjectLoader
from util import save
from util.draw import remove_right_top


def plot_paired_scatter(df, x, y, hue=None,
                        save_folder_name=None,
                        x_lim=None, y_lim=None,
                        x_label=None, y_label=None, tick_freq=0.1,
                        title=None,
                        diagonal=True,  # plot diagonal line
                        paired_test=True,
                        save_fig=False,
                        view_folder=False,
                        fig_ext='.png'):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ttest_rel
    import seaborn as sns

    # Parameters
    nb_row = 5
    nb_col = 2

    # Plot scatter with diagonal
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.suptitle(title, y=.9, fontsize=15)
    task_list = ['Predeafening', 'Postdeafening']

    for ind, task in enumerate(task_list):

        df_temp = df[df['taskName'] == task]
        ax = plt.subplot2grid((nb_row, nb_col), (1, ind), rowspan=3, colspan=1)

        if hue:  # color-code birds
            sns.scatterplot(x=x, y=y,
                            hue="birdID",
                            data=df_temp)
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

        else:
            sns.scatterplot(x=x, y=y, data=df_temp, color='k')

        if diagonal:
            ax.plot([0, 1], [0, 1], 'm--', transform=ax.transAxes, linewidth=1)

        if paired_test:
            ax_txt = plt.subplot2grid((nb_row, nb_col), (4, ind), rowspan=1, colspan=1)
            stat = ttest_rel(df_temp[x], df_temp[y])
            msg = f"t({len(df_temp[x].dropna()) - 2})=" \
                  f"{stat.statistic: 0.3f}, p={stat.pvalue: 0.3f}"
            ax_txt.text(0.25, 0, msg), ax_txt.axis('off')

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


def plot_by_day_per_syllable(fr_criteria=0, save_fig=False):
    """
    Plot daily pcc per syllable per bird
    Parameters
    ----------
    fr_criteria : 0 by default
        only plot the pcc for syllable having higher firing rates than criteria
    Returns
    -------

    """
    from database.load import ProjectLoader
    import matplotlib.pyplot as plt
    import seaborn as sns

    db = ProjectLoader().load_db()

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



def plot_regression(x, y, color='k', size=None, save_fig=False, fig_ext='.png', **kwargs):
    """Plot scatter plot between two continuous variables with its regression fit """

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.stats import pearsonr
    import statsmodels.api as sm

    def get_aic(x, y):
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        regr = OLS(y, add_constant(x)).fit()
        return regr.aic

    x = x.values.T
    y = y.values.T

    x_ind = x.argsort()
    x = x[x_ind]
    y = y[x_ind]

    # Plot figure
    fig = plt.figure(figsize=(7, 4))
    if 'fig_name' in kwargs:
        fig_name = kwargs['title']
    else:
        fig_name = 'Untitled'

    gs = gridspec.GridSpec(3, 4)

    # Plot scatter & regression
    ax = plt.subplot(gs[0:3, 0:3])
    plot = ax.scatter(x, y, c=color, s=size, edgecolors='k', cmap=plt.cm.hot_r)
    cbar = plt.colorbar(mappable=plot, ax=ax)
    #bar.set_clim(color.min(), color.max())
    cbar.set_label('Days after deafening')
    remove_right_top(ax)

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'x_label' in kwargs:
        ax.set_xlabel(kwargs['x_label'])
    if 'y_label' in kwargs:
        ax.set_ylabel(kwargs['y_label'])

    # Print out text results
    txt_xloc = 0
    txt_yloc = 0.8
    txt_inc = 0.2
    ax_txt = plt.subplot(gs[1:, -1])
    ax_txt.set_axis_off()

    # Regression analysis
    if 'regression_type' in kwargs:
        for fit in kwargs['regression_type']:

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            if fit == 'linear': # Linear regression
                corr, corr_pval = pearsonr(x.T[0], y.T[0])
                y_pred = LinearRegression().fit(x, y).predict(x)
                ax.plot(x, y_pred, color='r')
                ax_txt.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 4)}", fontsize=10)
                txt_yloc -= txt_inc
                ax_txt.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 4)}", fontsize=10)

                txt_yloc -= txt_inc
                aic_lin = get_aic(x, y_pred)
                ax_txt.text(txt_xloc, txt_yloc, f"aic (linear) = {round(aic_lin, 1)}", fontsize=10)

            if fit == 'quadratic': # Polynomial regression
                poly = PolynomialFeatures(degree=2)
                x_transformed = poly.fit_transform(x)
                model = sm.OLS(y, x_transformed).fit()
                y_pred = model.predict(x_transformed)
                fig.plot(x, y_pred, color='cyan')

                txt_yloc -= txt_inc
                aic_quad = get_aic(x, y_pred)
                ax_txt.text(txt_xloc, txt_yloc, f"aic (quad) = {round(aic_quad, 1)}", fontsize=10)

    if 'x_lim' in kwargs:
        ax.set_xlim(kwargs['x_lim'])
    if 'y_lim' in kwargs:
        ax.set_ylim(kwargs['y_lim'])

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
    else:
        plt.show()

