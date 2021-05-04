"""Compare pairwise cross-correlation (pcc) between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from results.plot import plot_bar_comparison
import seaborn as sns
from util import save
import numpy as np

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
df.set_index('id')

# Parameters
nb_row = 3
nb_col = 4
save_fig = True
fig_ext = '.png'

# Plot the results
fig, ax = plt.subplots(figsize=(10, 4))
plt.suptitle('Pairwise CC', y=.9, fontsize=20)

# Undir
df['pairwiseCorrUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['pairwiseCorrUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Undir', ylabel='PCC',
                    y_max=round(df['pairwiseCorrUndir'].max() * 10) / 10 + 0.1,
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Dir
df['pairwiseCorrDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['pairwiseCorrDir'], df['taskName'], hue_var=df['birdID'],
                    title='Dir', y_max=round(df['pairwiseCorrDir'].max() * 10) / 10 + 0.2,
                    col_order=("Predeafening", "Postdeafening"),
                    )
fig.tight_layout()


# Undir (paired comparisons)

pcc_mean_per_condition = df.groupby(['birdID','taskName'])['pairwiseCorrUndir'].mean().to_frame()
pcc_mean_per_condition.reset_index(inplace = True)

ax = plt.subplot2grid((nb_row, nb_col), (1, 2), rowspan=2, colspan=1)

ax = sns.pointplot(x='taskName', y='pairwiseCorrUndir', hue = 'birdID',
                   data=pcc_mean_per_condition,
                   order=["Predeafening", "Postdeafening"],
                   aspect=.5, hue_order = df['birdID'].unique().tolist(), scale = 0.7)

ax.spines['right'].set_visible(False),ax.spines['top'].set_visible(False)

title = 'Undir (Paired Comparison)'
title += '\n\n\n'

plt.title(title)
plt.xlabel(''), plt.ylabel('')
plt.ylim(0, 0.3), plt.xlim(-0.5, 1.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'PCC', fig_ext=fig_ext)
else:
    plt.show()

