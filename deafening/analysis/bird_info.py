"""Plot bird info"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.plot import plot_bar_comparison
import seaborn as sns
from util import save
import numpy as np
from util.draw import remove_right_top


# Parameters
nb_row = 3
nb_col = 4
circ_size = 35
save_fig = False
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()
# # SQL statement
song_df = db.to_dataframe(f"SELECT * FROM song")
song_df.set_index('id')
cluster_df = db.to_dataframe(f"SELECT * FROM cluster WHERE analysisOK=TRUE")
cluster_df.set_index('id')
bird_df = db.to_dataframe(f"SELECT * FROM bird")
bird_df.set_index('id')

# Plot recording days
# fig, ax = plt.subplots(figsize=(14, 6))
# plt.suptitle('Recording Days', y=.95, fontsize=15)
#
# ax = plt.subplot2grid((6, 1), (1, 0), rowspan=5, colspan=1)
# for ind, bird in enumerate(song_df['birdID'].unique()):
#     bird_df = song_df[(song_df['birdID'] == bird)]
#
#     ax.scatter(bird_df['taskSessionDeafening'], len(bird_df['taskSessionDeafening']) * [ind+1],
#                s=circ_size, color='k', facecolors='none')
#
#     cell_recording_days = cluster_df[cluster_df['birdID'] == bird]['taskSessionDeafening']
#     ax.scatter(cell_recording_days, len(cell_recording_days) * [ind+1],
#                s=circ_size, color='k')
#
# ax.legend(['Song only', 'Neural Recording'], loc='lower center', bbox_to_anchor=(0.9, 0.5))
# ax.set_yticks(range(1, len(song_df['birdID'].unique())+1))
# ax.set_yticklabels(song_df['birdID'].unique())
# ax.set_xlabel('Days', fontsize=14)
# ax.set_ylabel('Birds', fontsize=14)
# ax.axvline(x=0, color='b', ls='--', lw=0.8)
#
# fig.tight_layout()
# remove_right_top(ax)
# plt.show()


# Plot across age (dph)
fig, ax = plt.subplots(figsize=(25, 6))
plt.suptitle('Age', y=.95, fontsize=15)

ax = plt.subplot2grid((6, 1), (1, 0), rowspan=5, colspan=1)
ax.set_yticks(range(1, len(song_df['birdID'].unique())+1))
ax.set_yticklabels(song_df['birdID'].unique())
line_offsets = np.arange(0.5, len(song_df['birdID'].unique()))
surgery_age = np.array([], dtype=int)
for ind, bird in enumerate(song_df['birdID'].unique()):
    temp_df = song_df[(song_df['birdID'] == bird)]
    for age in temp_df['dph']:
        ax.scatter(age, ind+1,
                   s=circ_size, color='k', facecolors='none')

    ages = cluster_df[cluster_df['birdID'] == bird]['dph']
    for age in ages:
        ax.scatter(age, ind+1,
                   s=circ_size, color='k')

    surgery_age = bird_df.query(f"birdID == '{bird}'")['dphDeafening'].to_list()

    if surgery_age:
        print(bird, surgery_age)
        ax.eventplot(surgery_age, colors='r', lineoffsets=[ind+1],
                     linelengths=1, linewidths=3, orientation='horizontal')

ax.set_xlabel('Age (dph)', fontsize=14)
ax.set_ylabel('Birds', fontsize=14)
ax.set_ylim([0.5, 10.5])
fig.tight_layout()
remove_right_top(ax)
plt.show()

