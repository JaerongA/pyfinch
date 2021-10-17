"""
Analyze correlation between pairwise cross-correlation and song features
"""

from database.load import ProjectLoader
from deafening.plot import plot_regression

# Parameters
nb_row = 3
nb_col = 2

nb_note_crit = 10
fr_crit = 10

save_fig = False
view_folder = True  # open the folder where the result figures are saved
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()
# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
#         f"nbNoteUndir >={nb_note_crit} AND " \
#         f"taskName='Postdeafening'"
query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
        f"nbNoteUndir >={nb_note_crit}"
df = db.to_dataframe(query)

title = f'Undir FR >= {fr_crit} & # of Notes >= {nb_note_crit}'
x_label = 'EV'
y_label = 'PCC'
x_lim = [0, 0.035]
y_lim = [-0.05, 0.25]

# x = df['entropyUndir']
# x = df['spectroTempEntropyUndir']
x = df['entropyVarUndir']
y = df['pccUndir']
size = df['taskSessionPostDeafening']

plot_regression(x, y,
                # size=size,
                title=title,
                x_label=x_label, y_label=y_label,
                x_lim=x_lim,
                y_lim=y_lim,
                fig_name='PCC vs Entropy (Undir)',
                fr_criteria=fr_crit,
                save_fig=save_fig,
                fig_ext=fig_ext,
                regression_type=['linear']
                )

