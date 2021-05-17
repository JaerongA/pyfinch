"""
By Jaerong
plot raster & peth per syllable
"""

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from analysis.parameters import *
from analysis.spike import *
from database.load import DBInfo, ProjectLoader
from util import save
from util.draw import *
from analysis.parameters import pre_motor_win_size

# parameters
rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = 0.5  # text height
font_size = 12
marker_size = 0.4  # for spike count
nb_note_crit = 10  # minimum number of notes for analysis

norm_method = None
fig_ext = '.png'  # .png or .pdf
update = False  # Set True for recreating a cache file
save_fig = True
update_db = True  # save results to DB
time_warp = True  # spike time warping

# Load database
# SQL statement
# Create a new database (syllable)
db = ProjectLoader().load_db()
# query = "SELECT * FROM cluster WHERE analysisOK=1"
query = "SELECT * FROM cluster WHERE id=6"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format

    # Load class object
    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object

    import copy

    syllables = ''.join(ci.syllables)
    onsets = np.hstack(ci.onsets)
    offsets = np.hstack(ci.offsets)
    durations = np.hstack(ci.durations)
    contexts = ''

    for i in range(len(ci.contexts)):  # concatenate contexts
        contexts += ci.contexts[i] * len(ci.syllables[i])

    # Loop through note
    for note in cluster_db.songNote:

        ind = np.array(find_str(syllables, note))  # note indices
        if not ind.size:  # if a note does not exist
            continue
        note_onsets = np.asarray(list(map(float, onsets[ind])))
        note_offsets = np.asarray(list(map(float, offsets[ind])))
        note_durations = np.asarray(list(map(float, durations[ind])))
        note_contexts = ''.join(np.asarray(list(contexts))[ind])
        note_median_dur = np.median(note_durations, axis=0)
        nb_note = len(ind)

        spk_ts = np.hstack(ci.spk_ts)

        note_spk_ts_list = []
        for onset, offset in zip(note_onsets, note_offsets):
            note_spk_ts_list.append(spk_ts[np.where((spk_ts >= onset - pre_motor_win_size) & (spk_ts <= offset))])

        # Perform piecewise linear warping
        note_spk_ts_warped_list = []

        for onset, duration, spk_ts in zip(note_onsets, note_durations, note_spk_ts_list):
            spk_ts_new = copy.deepcopy(spk_ts)
            ratio = note_median_dur / duration
            offset = 0
            origin = 0

            spk_ts_temp, ind = spk_ts[spk_ts >= onset], np.where(spk_ts >= onset)

            spk_ts_temp = ((ratio * (spk_ts_temp - onset)) + origin) + onset
            np.put(spk_ts_new, ind, spk_ts_temp)  # replace original spk timestamps with warped timestamps
            note_spk_ts_warped_list.append(spk_ts_new)

        # Nb_note per context
        nb_note = {}
        for context in ['U', 'D']:
            nb_note[context] = len(find_str(note_contexts, context))

        # Get note firing rates (includes pre-motor window) per context
        note_spk = {}
        note_fr = {}
        for context1 in ['U', 'D']:
            if nb_note[context1] >= nb_note_crit:
                note_spk[context1] = sum(
                    [len(spk) for context2, spk in zip(note_contexts, note_spk_ts_list) if context2 == context1])
                note_fr[context1] = round(note_spk[context1] / (
                            (note_durations[find_str(note_contexts, context1)] + pre_motor_win_size).sum() / 1E3), 3)
            else:
                note_fr[context1] = np.nan

        # Skip if there are not enough motifs per condition
        if np.prod([nb[1] < nb_note_crit for nb in nb_note.items()]):
            print("Not enough notes")
            continue

        # Plot spectrogram & peri-event histogram (Just the first rendition)

        # Note start and end
        start = note_onsets[0] - peth_parm['buffer']
        end = note_offsets[0] + peth_parm['buffer']
        duration = note_offsets[0] - note_onsets[0]

        # Get spectrogram
        audio = AudioData(path, update=update).extract([start, end])  # audio object
        audio.spectrogram(freq_range=freq_range)

        # Plot figure
        fig = plt.figure(figsize=(6, 10))
        fig.set_tight_layout(False)
        note_name = ci.name + '-' + note
        if time_warp:
            fig_name = note_name + '  (time-warped)'
        else:
            fig_name = note_name + '  (non-warped)'
        plt.suptitle(fig_name, y=.93, fontsize=11)
        gs = gridspec.GridSpec(17, 5)
        gs.update(wspace=0.025, hspace=0.05)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3, 0:5])
        audio.spect_time = audio.spect_time - audio.spect_time[0] - peth_parm['buffer']  # starts from zero
        ax_spect.pcolormesh(audio.spect_time, audio.spect_freq, audio.spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_xlim(-peth_parm['buffer'], duration + peth_parm['buffer'])
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0, 0:5], sharex=ax_spect)
        onset = 0  # start from 0
        offset = onset + duration

        # Mark syllables
        rectangle = plt.Rectangle((onset, rec_yloc), duration, 0.2,
                                  linewidth=1, alpha=0.5, edgecolor='k',
                                  facecolor=note_color['Motif'][find_str(cluster_db.songNote, note)[0]])
        ax_syl.add_patch(rectangle)
        ax_syl.text((onset + (offset - onset) / 2), text_yloc, note, size=font_size)
        ax_syl.axis('off')

        # Calculate spectral entropy per time bin
        ax_se = ax_spect.twinx()
        se = audio.get_spectral_entropy()
        ax_se.plot(audio.spect_time, se, 'k')
        ax_se.set_ylim(0, 1)
        plt.show()

    break