from analysis.parameters import peth_parm, tick_length, tick_width, note_color, nb_note_crit
from analysis.spike import ClusterInfo
from database.load import DBInfo, ProjectLoader
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from util import save
from util.draw import remove_right_top
import warnings

from util.functions import myround

warnings.filterwarnings('ignore')


def _get_sorted_notes(target_note) -> str:
    """Get sequence of notes based on its median duration"""
    note_median_durs = np.array([], dtype=np.float32)
    for note in target_note:
        ni = ci.get_note_info(note)
        note_median_durs = np.append(note_median_durs, ni.median_dur)

    # Sort the note sequence by its duration
    sorted_notes = list(dict(sorted(list(zip(target_note, note_median_durs)), key=lambda x: x[-1])).keys())
    return ''.join(sorted_notes)


def _sort_trials_by_dur_per_note(NoteInfo):
    """
    Sort trials based on duration per note
    Parameters
    ----------
    NoteInfo : class object

    Returns
    -------
    contexts : str
    notes : str
    spk_ts : list
    onsets : arr
    durations : arr
    """
    note_all = ''  # all song notes in the session

    # Extract the info
    note_all += NoteInfo.note * len(NoteInfo.contexts)

    # Zip lists & filter & sort
    zipped_list = list(zip(NoteInfo.contexts, note_all, NoteInfo.spk_ts, NoteInfo.onsets, NoteInfo.durations))
    zipped_list = list(filter(lambda x: x[0] == context, zipped_list))  # filter context
    if sort_by_duration:
        zipped_list = sorted(zipped_list, key=lambda x: x[-1])  # sort by duration

    unzipped_object = zip(*zipped_list)
    contexts, notes, spk_ts, onsets, durations = list(unzipped_object)
    return ''.join(contexts), ''.join(notes), list(spk_ts), np.asarray((onsets)), np.asarray((durations))


# parameters
pre_evt_buffer = 50  # before syllable onset (in ms)
post_evt_buffer = 300  # after syllable onset (in ms)
bin_size = 5  # in ms
nb_bins = abs(pre_evt_buffer) + (2 * abs(post_evt_buffer))
target_note = None  # set to None to run across all syllables
marker_size = 2  # to mark the syllable offset  (5 for single syllable, 2 for all syllables)
context = 'U'
sort_by_duration = True
sort_by_syllable = True  # if True, plot syllables with the shortest duration first, if False, plot sorted by duration regardless of its identity

# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"

# Load database
db = ProjectLoader().load_db()
# SQL statement
db.execute(query)

cluster_db = DBInfo(db.cur.fetchall()[0])
name, path = cluster_db.load_cluster_db()
unit_nb = int(cluster_db.unit[-2:])
channel_nb = int(cluster_db.channel[-2:])
format = cluster_db.format
motif = cluster_db.motif
color_map = {note: color for note, color in zip(cluster_db.songNote, note_color['Motif'])}
# color_map = {note : color  for note, color in zip(cluster_db.introNotes + cluster_db.songNote + cluster_db.calls, note_color['Intro'] + note_color['Motif'] +  note_color['Call'] )}
# color_map = {note : color  for note, color in zip(cluster_db.introNotes + cluster_db.calls, note_color['Intro'] +  note_color['Call'] )}

# Load class object
ci = ClusterInfo(path, channel_nb, unit_nb, format, name)  # cluster object

note_all = ''  # all song notes in the session
note_contexts = ''
note_spks = []
note_onsets = np.array([], dtype='float32')
note_durations = np.array([], dtype='float32')
peth = np.array([], dtype=np.float32)
if not target_note:
    # target_note = cluster_db.introNotes + cluster_db.songNote + cluster_db.calls
    # target_note = cluster_db.introNotes + cluster_db.calls
    target_note = cluster_db.songNote

if sort_by_syllable:
    # Sort the notes based on median duration
    target_note = _get_sorted_notes(target_note)

    for ind, note in enumerate(target_note):
        ni = ci.get_note_info(note, pre_buffer=pre_evt_buffer, post_buffer=post_evt_buffer)
        contexts, notes, spk_ts, onsets, durations = _sort_trials_by_dur_per_note(ni)

        note_contexts += contexts
        note_all += notes
        note_spks.extend(spk_ts)
        note_onsets = np.append(note_onsets, onsets)
        note_durations = np.append(note_durations, durations)
        pi = ni.get_note_peth(time_warp=False, pre_evt_buffer=pre_evt_buffer, duration=post_evt_buffer,
                              bin_size=bin_size, nb_bins=nb_bins)
        if ind == 0:
            peth = pi.peth[context]
        else:
            peth = np.concatenate([peth, pi.peth[context]], axis=0)

    zipped_list = list(zip(note_contexts, note_all, note_spks, note_onsets, note_durations))

else:
    for ind, note in enumerate(target_note):
        ni = ci.get_note_info(note, pre_buffer=pre_evt_buffer, post_buffer=post_evt_buffer)
        note_all += ni.note * len(ni.contexts)
        note_contexts += ni.contexts
        note_spks.extend(ni.spk_ts)
        note_onsets = np.append(note_onsets, ni.onsets)
        note_durations = np.append(note_durations, ni.durations)
        pi = ni.get_note_peth(time_warp=False, pre_evt_buffer=pre_evt_buffer, duration=post_evt_buffer,
                              bin_size=bin_size, nb_bins=nb_bins)
        if ind == 0:
            peth = pi.peth[context]
        else:
            peth = np.concatenate([peth, pi.peth[context]], axis=0)

    # Zip lists & filter & sort
    zipped_list = list(zip(note_contexts, note_all, note_spks, note_onsets, note_durations))
    zipped_list = list(filter(lambda x: x[0] == context, zipped_list))  # filter context
    if sort_by_duration:
        zipped_list = sorted(zipped_list, key=lambda x: x[-1])  # sort by duration


# Plot figure
fig = plt.figure(figsize=(6, 8), dpi=600)
fig.set_tight_layout(False)

fig_name = f"{ci.name} ({target_note}) - {context}"
plt.suptitle(fig_name, y=.93, fontsize=10)
gs = gridspec.GridSpec(8, 7)
gs.update(wspace=0.025, hspace=0.05)

# Plot raster
ax_raster = plt.subplot(gs[0:6, 0:5])
line_offsets = np.arange(0.5, note_contexts.count(context))

for note_ind, (_, note, spk_ts, onset, duration) in enumerate(zipped_list):
    spk = spk_ts - onset
    ax_raster.eventplot(spk, colors='k', lineoffsets=line_offsets[note_ind],
                        linelengths=tick_length, linewidths=tick_width, orientation='horizontal')
    ax_raster.plot(duration, line_offsets[note_ind],
                   marker='o', color=color_map[note], markersize=marker_size, alpha=0.5, label=note)

ax_raster.axvline(x=0, linewidth=1, color='r', label='syllable onset', ls='--')

def _print_legend():
    # print out legend
    from matplotlib.lines import Line2D
    legend_marker = [Line2D([0], [0], color='r', ls='--', lw=1.5)]
    legend_label = ['onset']

    for note in target_note:
        legend_marker.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[note], markersize=marker_size * 3))
        legend_label.append(note)

    ax_raster.legend(legend_marker, legend_label,
                     loc='center left', bbox_to_anchor=(1.05, 0.5),
                     prop={'size': 8}, frameon=False)
_print_legend()


remove_right_top(ax_raster)
ax_raster.set_ylim([0, note_contexts.count(context) + 2])
ax_raster.set_xlim([-pre_evt_buffer, post_evt_buffer])
ax_raster.set_ylabel('Renditions')

# Plot PETH
ax_peth = plt.subplot(gs[7, 0:5], sharex=ax_raster)
if len(target_note) > 1:
    ax_peth.bar(pi.time_bin, peth.sum(axis=0), color='k', width=bin_size)
else:
    ax_peth.bar(pi.time_bin, peth.sum(axis=0), color=color_map[note], width=bin_size)
remove_right_top(ax_peth)
ax_peth.axvline(x=0, linewidth=1, color='r', ls='--')
ax_peth.set_ylim(0, myround(ax_peth.get_ylim()[1], base=10))
ax_peth.set_xlabel('Time (ms)')
ax_peth.set_ylabel('# of Spk')


# Print out corr metrics on the figure
txt_xloc = -0.5
txt_yloc = 1.5
txt_inc = 0.5  # y-distance between texts within the same section

pcc_undir = pcc_dir = max_cross_corr = peak_latency = np.nan
db.execute(f"SELECT crossCorrMax, peakLatency from song_fr_cross_corr WHERE clusterID= {cluster_db.id}")
for row in db.cur.fetchall():
    max_cross_corr, peak_latency = row[0], row[1]

# Run raster.py if pcc table is missing
db.execute(f"SELECT pccUndir, pccDir from pcc WHERE clusterID= {cluster_db.id}")
for row in db.cur.fetchall():
    pcc_undir, pcc_dir = row[0], row[1]

ax_txt = plt.subplot(gs[1, -1])
ax_txt.set_axis_off()
ax_txt.text(txt_xloc, txt_yloc, f"Cross-corr max = {max_cross_corr : 0.3f}", fontsize=8)
txt_yloc -= txt_inc

ax_txt.text(txt_xloc, txt_yloc, f"Peak latency = {int(peak_latency)} (ms)", fontsize=8)
txt_yloc -= txt_inc
if context == 'U':
    ax_txt.text(txt_xloc, txt_yloc, f"Motif PCC = {pcc_undir : .3f}", fontsize=8)
else:
    ax_txt.text(txt_xloc, txt_yloc, f"Motif PCC = {pcc_dir : .3f}", fontsize=8)
txt_yloc -= txt_inc
fig.tight_layout()
plt.show()


