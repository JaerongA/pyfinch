
from analysis.parameters import peth_parm, freq_range, peth_parm, tick_length, tick_width, note_color, nb_note_crit
from analysis.spike import ClusterInfo, MotifInfo, AudioData
from database.load import DBInfo, ProjectLoader, create_db
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from util import save
from util.functions import myround
from util.draw import remove_right_top
import warnings
warnings.filterwarnings('ignore')

# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"

# Load database
db = ProjectLoader().load_db()
# SQL statement
db.execute(query)

# parameters
pre_buffer = -100  # in ms
post_buffer = 500  # in ms

cluster_db = DBInfo(db.cur.fetchall()[0])
name, path = cluster_db.load_cluster_db()
unit_nb = int(cluster_db.unit[-2:])
channel_nb = int(cluster_db.channel[-2:])
format = cluster_db.format
motif = cluster_db.motif

# Load class object
        # Load class object
ci = ClusterInfo(path, channel_nb, unit_nb, format, name)  # cluster object
note = 'a'
ni = ci.get_note_info(note, pre_buffer=pre_buffer, post_buffer=post_buffer)

# Plot figure
fig = plt.figure(figsize=(7, 10), dpi=500)
fig.set_tight_layout(False)
note_name = ci.name + '-' + note
if time_warp:
    fig_name = note_name + '  (time-warped)'
else:
    fig_name = note_name + '  (non-warped)'
plt.suptitle(fig_name, y=.93, fontsize=11)
gs = gridspec.GridSpec(17, 5)
gs.update(wspace=0.025, hspace=0.05)


# Plot raster
ax_raster = plt.subplot(gs[4:6, 0:5])
line_offsets = np.arange(0.5, sum(ni.nb_note.values()))
zipped_lists = zip(ni.contexts, ni.spk_ts, ni.onsets, ni.durations)

for note_ind, (context, spk_ts, onset, duration) in enumerate(zipped_lists):

    spk = spk_ts - onset
    # print(len(spk))
    # print("spk ={}, nb = {}".format(spk, len(spk)))
    # print('')
    ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[note_ind],
                        linelengths=tick_length, linewidths=tick_width, orientation='horizontal')