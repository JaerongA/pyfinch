"""Compare spike fano factor between different conditions"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
import numpy as np
from util import save

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster")
df.set_index('id')

# Parameters
nb_row = 3
nb_col = 2
save_fig = False
fig_ext = '.png'

# Plot the results
fig, ax = plt.subplots(figsize=(6, 4))
plt.suptitle('Fano Factor', y=.9, fontsize=20)

# Undir
df['fanoSpkCountUndir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
ax = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['fanoSpkCountUndir'], df['taskName'], hue_var=df['birdID'],
                    title='Undir', ylabel='Fano Factor',
                    y_max=round(df['fanoSpkCountUndir'].max()) + 1,
                    col_order=("Predeafening", "Postdeafening"),
                    )

# Dir
df['fanoSpkCountDir'].replace('', np.nan, inplace=True)  # replace empty values with nans to prevent an error
ax = plt.subplot2grid((nb_row, nb_col), (1, 1), rowspan=2, colspan=1)
plot_bar_comparison(ax, df['fanoSpkCountDir'], df['taskName'], hue_var=df['birdID'],
                    title='Dir',
                    y_max=round(df['fanoSpkCountDir'].max()) + 1,
                    col_order=("Predeafening", "Postdeafening"),
                    )
fig.tight_layout()


# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'FanoFactor', fig_ext=fig_ext)
else:
    plt.show()

