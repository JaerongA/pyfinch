"""Plotting functions & classes specific to the deafening project"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
import seaborn as sns
from util import save
from util.draw import remove_right_top
import warnings

warnings.filterwarnings('ignore')
save_folder_name = 'Results'

def get_nb_cluster(ax):
    """
    Print out the number of usable clusters in a table for each condition
    Parameters
    ----------
    ax : axis object
    """
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
          loc="center", colWidths=[0.8, 0.8, 0.8], cellLoc='center')


def plot_scatter_diagonal(df, x, y, hue=None,
                          save_folder_name=None,
                          x_lim=None, y_lim=None,
                          x_label=None, y_label=None, tick_freq=0.1,
                          title=None,
                          diagonal=True,  # plot diagonal line
                          paired_test=True,
                          save_fig=False,
                          view_folder=False,
                          fig_ext='.png'):
    import numpy as np
    from scipy.stats import ttest_rel

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
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
        save.save_fig(fig, save_path, title, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()


def plot_by_day_per_syllable(fr_criteria=0,
                             save_fig=False,
                             fig_ext='.png'):
    """
    Plot daily pcc per syllable per bird
    Parameters
    ----------
    fr_criteria : 0 (default)
        only plot the pcc for syllable having higher firing rates than criteria
    save_fig : bool
        save figure
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


def plot_regression(x, y, color='k', size=None, save_fig=False, fig_ext='.png',
                    view_folder=True,
                    **kwargs):
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

    # x = x.values.T
    # y = y.values.T
    #
    # x_ind = x.argsort()
    # x = x[x_ind]
    # y = y[x_ind]

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
    if color != 'k':
        cbar = plt.colorbar(mappable=plot, ax=ax)
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

            if fit == 'linear':  # Linear regression
                corr, corr_pval = pearsonr(x.T[0], y.T[0])
                y_pred = LinearRegression().fit(x, y).predict(x)
                ax.plot(x, y_pred, color='r')
                ax_txt.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 4)}", fontsize=10)
                txt_yloc -= txt_inc
                ax_txt.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 4)}", fontsize=10)

                txt_yloc -= txt_inc
                aic_lin = get_aic(x, y_pred)
                ax_txt.text(txt_xloc, txt_yloc, f"aic (linear) = {round(aic_lin, 1)}", fontsize=10)

            if fit == 'quadratic':  # Polynomial regression
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
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()


def plot_bar_comparison(ax, dependent_var, group_var, hue_var=None,
                        title='', x_label=None, y_label=None,
                        col_order=None,
                        y_lim=None,
                        jitter=0.1, alpha=0.5,
                        run_stats=True, stat_txt_size=10,
                        legend_ok=False,
                        **kwargs
                        ):
    import numpy as np
    from scipy import stats
    from util.stats import get_sig

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
    # title += '\n\n\n'
    # plt.title(title)
    ax.set_xlabel(x_label), ax.set_ylabel(y_label)

    # Add stat comparisons (2-sample t-test)
    if run_stats:
        group1 = dependent_var[group_var == list(set(group_var))[0]].dropna()
        group2 = dependent_var[group_var == list(set(group_var))[1]].dropna()
        tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
        degree_of_freedom = len(group1) + len(group2) - 2

        sig = get_sig(pval)  # print out asterisk

        if sig == '***':  # mark significance
            msg2 = '$P$ < 0.001'
        else:
            msg2 = ('$P$ = {:.3f}'.format(pval))
        # plt.text((x1 + x2) * .5, y * 1.1, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
        msg1 = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
        # plt.text((x1 + x2) * .5, y * 1.2, msg, ha='center', va='bottom', color=col, size=stat_txt_size)
        ax.set_title(f"{title} \n\n {msg1} \n {msg2}", size=10)

    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    if 'xtick_label' in kwargs:
        ax.set_xticklabels(kwargs['xtick_label'])

    if legend_ok and hue_var is not None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif legend_ok == False and hue_var is not None:
        ax.get_legend().remove()


def plot_across_days(df, x, y,
                     x_lim=None, y_lim=None,
                     x_label=None, y_label=None,
                     title=None,
                     save_fig=False,
                     view_folder=True,
                     fig_ext='.png',
                     **kwargs
                     ):
    # Plot the results
    circ_size = 0.5
    bird_list = df['birdID'].unique()
    fig, axes = plt.subplots(2, 5, figsize=(21, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)

    if title:  # title
        plt.suptitle(title, y=.98, fontsize=18)
        axes = axes.ravel()

    if 'fig_name' in kwargs:
        fig_name = kwargs['title']
    else:
        fig_name = 'Untitled'

    for bird, ax_ind in zip(bird_list, range(len(bird_list))):

        df_temp = df.loc[df['birdID'] == bird]
        sns.lineplot(x=x, y=y,
                     data=df_temp, marker='o', color='k', mew=circ_size, ax=axes[ax_ind])
        remove_right_top(axes[ax_ind])
        axes[ax_ind].set_title(bird)

        if x_label:
            if ax_ind >= 5:
                axes[ax_ind].set_xlabel('Days from deafening')
            else:
                axes[ax_ind].set_xlabel('')

        if y_label:
            if (ax_ind == 0) or (ax_ind == 5):
                axes[ax_ind].set_ylabel(y_label)
            else:
                axes[ax_ind].set_ylabel('')

        axes[ax_ind].set_xlim(x_lim)
        axes[ax_ind].set_ylim(y_lim)
        axes[ax_ind].axvline(x=0, color='k', linestyle='dashed', linewidth=0.5)

    # Save figure
    if save_fig:
        save.save_fig(fig, kwargs['save_path'], fig_name, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()


def plot_across_days_per_note(df, x, y,
                              x_label=None,
                              y_label=None,
                              title=None, fig_name=None,
                              x_lim=None, y_lim=None,
                              vline=None,
                              hline=None,
                              view_folder=True,
                              save_fig=True,
                              save_path=None,
                              fig_ext='.png'
                              ):
    """
    Plot values across sessions per block & per bird
    Parameters
    ----------
    df : Dataframe
    x : str
        column name (independent variable)
    y : str
        column name (dependent variable)
    x_label : str
    y_label : str
    title : str
    fig_name : str
    x_lim : list
    y_lim : list
    vline : float
    hline : float
    view_folder : bool
    save_fig : bool
    save_path : Path
    fig_ext : str
    """

    circ_size = 1
    bird_list = df['birdID'].unique()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=.3, wspace=.2, top=0.9)
    if title:
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

        if x_lim:
            axes[ax_ind].set_xlim(x_lim)
        if y_lim:
            axes[ax_ind].set_ylim(y_lim)

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
                     fig_size=(5, 5),
                     view_folder=True,
                     fig_name=None,
                     save_fig=True,
                     save_path=None,
                     color_by_bird=True,
                     fig_ext='.png'
                     ):
    """
    Plot paired data
    data points connected by lines if paired
    each data point from one syllable or one bird
    """
    from scipy import stats
    from util.stats import get_sig

    if 'note' in df.columns:
        df_mean = df.groupby(['birdID', 'note', 'taskName']).mean()[y].reset_index()
    else:
        df_mean = df.groupby(['birdID', 'taskName']).mean()[y].reset_index()
    df_mean.dropna(inplace=True)

    # Connect paired data
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    current_palette = sns.color_palette()
    inc = 0

    if 'note' in df.columns:
        for bird in df_mean['birdID'].unique():
            color = current_palette[inc] if color_by_bird else 'k'
            for note in df_mean['note'].unique():
                temp_df = df_mean.loc[(df_mean['birdID'] == bird) & (df_mean['note'] == note)]
                if len(temp_df[x].unique()) == 2:  # paired value exists

                    ax = sns.pointplot(x=x, y=y,
                                       data=temp_df,
                                       order=["Predeafening", "Postdeafening"],
                                       aspect=.5, scale=0.7, color=color)
                elif len(temp_df[x].unique()) == 1:  # paired value doesn't exist
                    if temp_df[x].values == 'Predeafening':
                        ax.scatter(-0.2, temp_df[y].values, color=color)
                    if temp_df[x].values == 'Postdeafening':
                        ax.scatter(1.2, temp_df[y].values, color=color)
            inc += 1
    else:
        for bird in df_mean['birdID'].unique():
            color = current_palette[inc] if color_by_bird else 'k'
            temp_df = df_mean.loc[(df_mean['birdID'] == bird)]
            if len(temp_df[x].unique()) == 2:  # paired value exists

                ax = sns.pointplot(x=x, y=y,
                                   data=temp_df,
                                   order=["Predeafening", "Postdeafening"],
                                   aspect=.5, scale=0.7, color=color)
                inc += 1
            elif len(temp_df[x].unique()) == 1:  # paired value doesn't exist
                if temp_df[x].values == 'Predeafening':
                    ax.scatter(-0.2, temp_df[y].values, color=color)
                if temp_df[x].values == 'Postdeafening':
                    ax.scatter(1.2, temp_df[y].values, color=color)
                inc += 1

    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel(x_label), ax.set_ylabel(y_label)
    remove_right_top(ax)

    # 2 sample t-test (independent)
    group_var = df_mean[x]
    dependent_var = df_mean[y]
    group1 = dependent_var[group_var == list(set(group_var))[0]].dropna()
    group2 = dependent_var[group_var == list(set(group_var))[1]].dropna()
    tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
    degree_of_freedom = len(group1) + len(group2) - 2
    msg1 = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
    if pval < 0.001:  # mark significance
        msg2 = 'p < 0.001'
    else:
        msg2 = ('p = {:.3f}'.format(pval))
    msg = msg1 + ', ' + msg2
    sig = get_sig(pval)  # print out asterisk

    # rank-sum test (non-parametric independent 2-sample test)
    from scipy.stats import ranksums
    stat, pval = stats.ranksums(group1, group2)
    msg = f"ranksum p-val = {pval : .3f}"
    if save_fig:
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()


def plot_per_day_block(df, ind_var_name, dep_var_name,
                       title=None, y_label=None, y_lim=None,
                       plot_type='bar',  # {bar, box}
                       post_hoc=False,
                       fig_name='Untitled',
                       view_folder=True,
                       save_fig=True, save_folder_name=save_folder_name,
                       fig_ext='.png'
                       ):
    """Plot bar plot and violin plot for values per day block and run one-way ANOVA"""

    # Plot the results
    x = df[ind_var_name]
    y = df[dep_var_name]
    ind_groups = df[ind_var_name].sort_values().unique()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.suptitle(title, y=1, fontsize=12)

    if plot_type == 'bar':
        sns.barplot(x, y, ax=axes[0], facecolor=(1, 1, 1, 0),
                    linewidth=1,
                    errcolor=".2", edgecolor=".2", zorder=0)
    elif plot_type == 'box':
        sns.stripplot(x, y, ax=axes[0], color="k", jitter=0.1, size=4)
        sns.boxplot(x, y, ax=axes[0], width=0.5, color='w', showfliers=False)

    remove_right_top(axes[0])

    sns.violinplot(x, y, ax=axes[1], inner=None)
    sns.swarmplot(x, y, ax=axes[1], color="k", size=3)
    remove_right_top(axes[1])

    axes[0].set_xlabel('Day Block (10 days)'), axes[0].set_ylabel(y_label)
    axes[1].set_xlabel('Day Block (10 days)'), axes[1].set_ylabel('')
    if y_lim:
        axes[0].set_ylim(y_lim), axes[1].set_ylim(y_lim)

    if len(ind_groups) == 5:
        day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day 21-30', 'Day >= 31']
    elif len(ind_groups) == 4:
        day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day >= 21']
    axes[0].set_xticklabels(day_block_label_list, rotation=45)
    axes[1].set_xticklabels(day_block_label_list, rotation=45)

    # Run one-way ANOVA
    import scipy.stats as stats
    from util.stats import get_sig
    if len(ind_groups) == 5:
        f_val, p_val = stats.f_oneway(df[dep_var_name][df[ind_var_name] == 0].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 1].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 2].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 3].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 4].to_numpy()
                                      )
    elif len(ind_groups) == 4:
        f_val, p_val = stats.f_oneway(df[dep_var_name][df[ind_var_name] == 0].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 1].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 2].to_numpy(),
                                      df[dep_var_name][df[ind_var_name] == 3].to_numpy()
                                      )
    sig = get_sig(p_val)  # print out asterisk

    # Get degrees of freedom
    total_num = df[dep_var_name][df[ind_var_name]].shape[0]
    num_groups = df[ind_var_name].unique().shape[0]
    df_bn = df[ind_var_name].unique().shape[0] - 1
    df_wn = total_num - num_groups

    msg = f"""One-way ANOVA \n\n F({df_bn}, {df_wn})={f_val: 0.3f}, p={p_val: 0.3f} ({sig})"""
    axes[2].text(0, 0.5, msg), axes[2].axis('off')

    # Run post-hoc comparisons
    # The output values don't match those from Statview (This needs to be checked)
    if post_hoc:
        # Bonferroni post-hoc test
        import scikit_posthocs as sp
        p_map = sp.posthoc_ttest(df, val_col=dep_var_name, group_col=ind_var_name, p_adjust='bonferroni', sort=True)

        # Tukey/Kramer post-hoc test
        from statsmodels.stats.multicomp import MultiComparison
        mc = MultiComparison(df[dep_var_name], df[ind_var_name])
        print(mc.tukeyhsd())
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()


def plot_line_across_blocks(df_block, variable,
                            title=None,
                            y_lim=None, fig_size=(4, 3),
                            y_label=None,
                            fig_name='Untitled',
                            view_folder=True,
                            save_fig=False, save_folder_name=save_folder_name,
                            fig_ext='.png'
                            ):
    """
    Plot line plot with connecting lines across day block and run one-way ANOVA
    df_block should have a block column and block-averaged variable column
    """

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.suptitle(title, y=1, fontsize=12)
    sns.lineplot('block10days', variable, hue='birdID', marker='o', data=df_block, ax=ax)
    remove_right_top(ax)
    ax.set_xlabel(''), ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)

    if len(df_block['block10days'].unique()) == 4:
        day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day >= 21']
        ax.set_xticks([0, 1, 2, 3])

    elif len(df_block['block10days'].unique()) == 5:
        day_block_label_list = ['Predeafening', 'Day 4-10', 'Day 11-20', 'Day 21-30', 'Day >= 31']
        ax.set_xticks([0, 1, 2, 3, 4])

    ax.set_xticklabels(day_block_label_list, rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    # Run one-way ANOVA
    import scipy.stats as stats
    from util.stats import get_sig

    if len(df_block['block10days'].unique()) == 4:
        f_val, p_val = stats.f_oneway(
            df_block[variable][df_block['block10days'] == 0],
            df_block[variable][df_block['block10days'] == 1],
            df_block[variable][df_block['block10days'] == 2],
            df_block[variable][df_block['block10days'] == 3],
        )
    elif  len(df_block['block10days'].unique()) == 5:
        f_val, p_val = stats.f_oneway(
            df_block[variable][df_block['block10days'] == 0],
            df_block[variable][df_block['block10days'] == 1],
            df_block[variable][df_block['block10days'] == 2],
            df_block[variable][df_block['block10days'] == 3],
            df_block[variable][df_block['block10days'] == 4]
        )

    sig = get_sig(p_val)  # print out asterisk

    msg = f"""F ={f_val: 0.3f}, p ={p_val: 0.3f} (One-way ANOVA) ({sig})"""
    # print(msg)
    ax.text(0, -0.5, msg, transform=ax.transAxes)
    plt.tight_layout()
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
        save.save_fig(fig, save_path, fig_name, view_folder=view_folder, fig_ext=fig_ext)
    else:
        plt.show()


def pre_post_comparison(query,
                        x, y1, y2,
                        title=None,
                        run_stats=True,
                        y_lim=None,
                        fig_ext='.png',
                        save_fig=False,
                        update_cache=False):
    from database.load import ProjectLoader
    import matplotlib.pyplot as plt

    # Parameters
    nb_row = 3
    nb_col = 2

    # Load database
    db = ProjectLoader().load_db()
    # # SQL statement

    df = db.to_dataframe(query)
    # df.set_index('id')

    # Plot the results
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(title, y=.9, fontsize=20)

    # Undir
    ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df[y1], df[x], hue_var=df['birdID'],
                        title='Undir', y_label=y1,
                        col_order=("Predeafening", "Postdeafening"),
                        y_lim=y_lim,
                        run_stats=run_stats
                        )
    # Dir
    ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
    plot_bar_comparison(ax, df[y2], df[x], hue_var=df['birdID'],
                        title='Dir', y_label=y2,
                        col_order=("Predeafening", "Postdeafening"),
                        y_lim=y_lim,
                        run_stats=run_stats,
                        legend_ok=True
                        )
    fig.tight_layout()
    plt.show()
