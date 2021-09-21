"""Get the proportion of syllable having a significant PCC"""

from collections import defaultdict

import matplotlib.pyplot as plt

from database.load import ProjectLoader
from util import save
from util.draw import remove_right_top

# Parameters
nb_note_crit = 10  # minimum number of notes for analysis
fr_crit = 10  # in Hz
task_name = ['Predeafening', 'Postdeafening']

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM syllable_pcc_shuffle")
df.set_index('syllableID')

# Filter through criteria
df_undir = df[(df.nbNoteUndir >= nb_note_crit) & (df.frUndir >= fr_crit)]
df_dir = df[(df.nbNoteDir >= nb_note_crit) & (df.frDir >= fr_crit)]

# Parameters
save_fig = False
fig_ext = '.png'
peth_shuffle = {'shuffle_limit': [1, 5, 10, 15, 20],  # in ms
                'shuffle_iter': 100}  # bootstrap iterations

sig_prop = defaultdict(lambda: defaultdict(list))

for shuffle_limit in peth_shuffle['shuffle_limit']:

    for task in df['taskName'].unique():
        df_task_undir = df_undir[(df_undir.taskName == task)]
        df_task_dir = df_dir[(df_dir.taskName == task)]

        pcc_col_names_undir = f"pccUndirSig_{shuffle_limit}"
        pcc_col_names_dir = f"pccDirSig_{shuffle_limit}"

        sig_prop[task]['U'].append(
            df_task_undir[pcc_col_names_undir].sum() / df_task_undir[pcc_col_names_undir].count())
        sig_prop[task]['D'].append(df_task_dir[pcc_col_names_dir].sum() / df_task_dir[pcc_col_names_dir].count())

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, task in zip(axes, task_name):
    for ind, (shuffle_limit, u_prop, d_prop) in \
            enumerate(zip(peth_shuffle['shuffle_limit'], sig_prop[task]['U'], sig_prop[task]['D'])):
        ax.plot(shuffle_limit, u_prop, 'bo')
        ax.plot(shuffle_limit, d_prop, 'mo')

    ax.plot(peth_shuffle['shuffle_limit'], sig_prop[task]['U'], 'b')
    ax.plot(peth_shuffle['shuffle_limit'], sig_prop[task]['D'], 'm')

    ax.set_ylim([0.25, 1.05])
    ax.set_xlim([0, peth_shuffle['shuffle_limit'][-1] + 0.5])
    ax.set_title(task)
    ax.set_xlabel('Jitter (ms)')
    ax.set_xticks(peth_shuffle['shuffle_limit'])
    ax.set_xticklabels(peth_shuffle['shuffle_limit'])
    remove_right_top(ax)

    ax.legend(['U', 'D'], loc="lower right")

    if task == 'Predeafening':
        ax.set_ylabel('Proportion of syllables')

plt.show()

# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'PCC', fig_ext=fig_ext)
else:
    plt.show()
