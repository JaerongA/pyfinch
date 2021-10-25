"""
Analyze correlation between pairwise cross-correlation and song features
"""

from database.load import ProjectLoader
from deafening.plot import plot_regression

# Parameters
nb_note_crit = 10
fr_crit = 10
save_fig = False
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()

# query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
#         f"nbNoteUndir >={nb_note_crit}"

#query = f"SELECT * FROM syllable_pcc WHERE frUndir >= {fr_crit} AND " \
#        f"nbNoteUndir >={nb_note_crit} AND " \
#        f"taskName='Predeafening'"

query = f"""SELECT * FROM syllable_pcc 
WHERE frUndir >= {fr_crit} 
AND nbNoteUndir >={nb_note_crit} AND birdID!='o25w75' AND taskName='Postdeafening'"""

df = db.to_dataframe(query)

# Plot results
title = f'Undir FR >= {fr_crit} & # of Notes >= {nb_note_crit}'
color = df['taskSessionDeafening'].to_numpy()
size = df['taskSessionPostDeafening'].to_numpy()

# Entropy Variance
plot_regression(x = df['entropyVarUndir'].to_numpy(),
                y = df['pccUndir'].to_numpy(),
                # size=size,
                color=color,
                title=title,
                x_label='EV', y_label='PCC',
                x_lim=[0, 0.035],
                y_lim=[-0.05, 0.25],
                fig_name='PCC vs EV (Undir)',
                fr_criteria=fr_crit,
                save_fig=save_fig,
                fig_ext=fig_ext,
                regression_type=['linear']
                )

# Entropy
plot_regression(x = df['entropyUndir'].to_numpy(),
                y = df['pccUndir'].to_numpy(),
                # size=size,
                color=color,
                title=title,
                x_label='Entropy', y_label='PCC',
                x_lim=[0.2, 0.9],
                y_lim=[-0.05, 0.25],
                fig_name='PCC vs Entropy (Undir)',
                fr_criteria=fr_crit,
                save_fig=save_fig,
                fig_ext=fig_ext,
                regression_type=['linear']
                )

# Spectro-temporal entropy
plot_regression(x = df['spectroTempEntropyUndir'].to_numpy(),
                y = df['pccUndir'].to_numpy(),
                # size=size,
                color=color,
                title=title,
                x_label='SpectroTemporalEntropy', y_label='PCC',
                x_lim=[0.2, 0.8],
                y_lim=[-0.05, 0.25],
                fig_name='PCC vs STEntropy (Undir)',
                fr_criteria=fr_crit,
                save_fig=save_fig,
                fig_ext=fig_ext,
                regression_type=['linear']
                )
