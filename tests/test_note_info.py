import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from util.draw import *

from pyfinch.core import *
from pyfinch.core import pre_motor_win_size
from pyfinch.db.load import DBInfo, ProjectLoader

# parameters
rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = 0.5  # text height
font_size = 12
marker_size = 0.4  # for spike count
nb_note_crit = 10  # minimum number of notes for analysis

# Gauss parameter for PETH smoothing
norm_method = None
fig_ext = ".png"  # .png or .pdf
update = False  # Set True for recreating a cache file
save_fig = False
update_db = False  # save results to DB
time_warp = True  # spike time warping


# Load database
# SQL statement
# Create a new database (syllable)
db = ProjectLoader().load_db()
# query = "SELECT * FROM cluster WHERE analysisOK=1"
query = "SELECT * FROM cluster WHERE id=96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    motif = cluster_db.motif

    # Load class object
    ci = ClusterInfo(
        path, channel_nb, unit_nb, format, name, update=update
    )  # cluster object

    # Loop through note
    for note in cluster_db.songNote:

        ni = ci.get_note_info(note)

        # Skip if there are not enough motifs per condition
        if np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()]):
            print("Not enough notes")
            continue

        # Plot spectrogram & peri-event histogram (Just the first rendition)
        # Note start and end
        start = ni.onsets[0] - peth_parm["buffer"]
        end = ni.offsets[0] + peth_parm["buffer"]
        duration = ni.durations[0]

        # Get spectrogram
        audio = AudioData(path, update=update).extract([start, end])  # audio object
        audio.spectrogram(freq_range=freq_range)

        # Plot figure
        fig = plt.figure(figsize=(6, 10))
        fig.set_tight_layout(False)
        note_name = ci.name + "-" + note
        if time_warp:
            fig_name = note_name + "  (time-warped)"
        else:
            fig_name = note_name + "  (non-warped)"
        plt.suptitle(fig_name, y=0.93, fontsize=11)
        gs = gridspec.GridSpec(17, 5)
        gs.update(wspace=0.025, hspace=0.05)

        # Plot spectrogram
        ax_spect = plt.subplot(gs[1:3, 0:5])
        audio.spect_time = (
            audio.spect_time - audio.spect_time[0] - peth_parm["buffer"]
        )  # starts from zero
        ax_spect.pcolormesh(
            audio.spect_time,
            audio.spect_freq,
            audio.spect,  # data
            cmap="hot_r",
            norm=colors.SymLogNorm(linthresh=0.05, linscale=0.03, vmin=0.5, vmax=100),
        )

        remove_right_top(ax_spect)
        ax_spect.set_xlim(-peth_parm["buffer"], duration + peth_parm["buffer"])
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel("Frequency (Hz)", fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)

        # Plot syllable duration
        ax_syl = plt.subplot(gs[0, 0:5], sharex=ax_spect)
        onset = 0  # start from 0
        offset = onset + duration

        # Mark syllables
        rectangle = plt.Rectangle(
            (onset, rec_yloc),
            duration,
            0.2,
            linewidth=1,
            alpha=0.5,
            edgecolor="k",
            facecolor=note_color["Motif"][find_str(cluster_db.songNote, note)[0]],
        )
        ax_syl.add_patch(rectangle)
        ax_syl.text((onset + (offset - onset) / 2), text_yloc, note, size=font_size)
        ax_syl.axis("off")

        # Plot raster
        ax_raster = plt.subplot(gs[4:6, 0:5], sharex=ax_spect)
        line_offsets = np.arange(0.5, sum(ni.nb_note.values()))
        # ni.spk_ts_warped = note_spk_ts_warped_list
        if time_warp:
            zipped_lists = zip(ni.contexts, ni.spk_ts_warp, ni.onsets)
        else:
            zipped_lists = zip(ni.contexts, ni.spk_ts, ni.onsets)

        pre_context = ""  # for marking  context change
        context_change = np.array([])

        for note_ind, (context, spk_ts, onset) in enumerate(zipped_lists):

            spk = spk_ts - onset
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(
                spk,
                colors="k",
                lineoffsets=line_offsets[note_ind],
                linelengths=tick_length,
                linewidths=tick_width,
                orientation="horizontal",
            )

            # Demarcate the note
            if time_warp:
                note_duration = ni.median_dur
            else:
                note_duration = ni.durations[note_ind]

            rectangle = plt.Rectangle(
                (0, note_ind),
                note_duration,
                rec_height,
                fill=True,
                linewidth=1,
                alpha=0.15,
                facecolor=note_color["Motif"][find_str(cluster_db.songNote, note)[0]],
            )
            ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=note_ind, color="k", ls="-", lw=0.3)
                context_change = np.append(context_change, (note_ind))
                if pre_context:
                    ax_raster.text(
                        ax_raster.get_xlim()[1] + 0.2,
                        ((context_change[-1] - context_change[-2]) / 3)
                        + context_change[-2],
                        pre_context,
                        size=6,
                    )
            pre_context = context

        # Demarcate the last block
        ax_raster.text(
            ax_raster.get_xlim()[1] + 0.2,
            ((ax_raster.get_ylim()[1] - context_change[-1]) / 3) + context_change[-1],
            pre_context,
            size=6,
        )

        ax_raster.set_yticks([0, sum(ni.nb_note.values())])
        ax_raster.set_yticklabels([0, sum(ni.nb_note.values())])
        ax_raster.set_ylim([0, sum(ni.nb_note.values())])
        ax_raster.set_ylabel("Trial #", fontsize=font_size)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        remove_right_top(ax_raster)

        # Plot sorted raster
        ax_raster = plt.subplot(gs[7:9, 0:5], sharex=ax_spect)

        # Sort trials based on context
        sort_ind = np.array(
            [
                i[0]
                for i in sorted(
                    enumerate(ni.contexts), key=lambda x: x[1], reverse=True
                )
            ]
        )
        contexts_sorted = np.array(list(ni.contexts))[sort_ind].tolist()
        # ni.onsets = note_onsets
        onsets_sorted = np.array(ni.onsets)[sort_ind].tolist()
        if time_warp:
            spk_ts_sorted = np.array(ni.spk_ts_warp)[sort_ind].tolist()
        else:
            # ni.spk_ts = note_spk_ts_list
            spk_ts_sorted = np.array(ni.spk_ts)[sort_ind].tolist()

        zipped_lists = zip(contexts_sorted, spk_ts_sorted, onsets_sorted)

        pre_context = ""  # for marking  context change
        context_change = np.array([])

        for note_ind, (context, spk_ts, onset) in enumerate(zipped_lists):

            # Plot rasters
            spk = spk_ts - onset
            # print(len(spk))
            # print("spk ={}, nb = {}".format(spk, len(spk)))
            # print('')
            ax_raster.eventplot(
                spk,
                colors="k",
                lineoffsets=line_offsets[note_ind],
                linelengths=tick_length,
                linewidths=tick_width,
                orientation="horizontal",
            )

            # Demarcate the note
            if time_warp:
                note_duration = ni.median_dur
            else:
                note_duration = ni.durations[note_ind]

            rectangle = plt.Rectangle(
                (0, note_ind),
                note_duration,
                rec_height,
                fill=True,
                linewidth=1,
                alpha=0.15,
                facecolor=note_color["Motif"][find_str(cluster_db.songNote, note)[0]],
            )
            ax_raster.add_patch(rectangle)

            # Demarcate song block (undir vs dir) with a horizontal line
            if pre_context != context:
                ax_raster.axhline(y=note_ind, color="k", ls="-", lw=0.3)
                context_change = np.append(context_change, (note_ind))
                if pre_context:
                    ax_raster.text(
                        ax_raster.get_xlim()[1] + 0.2,
                        ((context_change[-1] - context_change[-2]) / 3)
                        + context_change[-2],
                        pre_context,
                        size=6,
                    )
            pre_context = context

        # Demarcate the last block
        ax_raster.text(
            ax_raster.get_xlim()[1] + 0.2,
            ((ax_raster.get_ylim()[1] - context_change[-1]) / 3) + context_change[-1],
            pre_context,
            size=6,
        )

        ax_raster.set_yticks([0, sum(ni.nb_note.values())])
        ax_raster.set_yticklabels([0, sum(ni.nb_note.values())])
        ax_raster.set_ylim([0, sum(ni.nb_note.values())])
        ax_raster.set_ylabel("Trial #", fontsize=font_size)
        ax_raster.set_title("sorted raster", size=font_size)
        plt.setp(ax_raster.get_xticklabels(), visible=False)
        remove_right_top(ax_raster)

        # Get firing rates
        pi = ni.get_note_peth()
        pi.get_fr()  # get firing rates

        # Plot mean firing rates
        ax_peth = plt.subplot(gs[10:12, 0:5], sharex=ax_spect)
        for context, fr in pi.mean_fr.items():
            if context == "U":
                ax_peth.plot(pi.time_bin, fr, "b", label=context)
            elif context == "D":
                ax_peth.plot(pi.time_bin, fr, "m", label=context)

        plt.legend(
            loc="center left", bbox_to_anchor=(0.98, 0.5), prop={"size": 6}
        )  # print out legend
        ax_peth.set_ylabel("FR", fontsize=font_size)

        fr_ymax = myround(round(ax_peth.get_ylim()[1], 3), base=5)
        ax_peth.set_ylim(0, fr_ymax)
        plt.yticks([0, ax_peth.get_ylim()[1]], [str(0), str(int(fr_ymax))])

        # Mark the baseline firing rates
        if "baselineFR" in row.keys() and cluster_db.baselineFR:
            ax_peth.axhline(y=row["baselineFR"], color="k", ls="--", lw=0.5)

        # Mark end of the motif
        ax_peth.axvline(x=0, color="k", ls="--", lw=0.5)
        ax_peth.axvline(x=ni.median_dur, color="k", lw=0.5)
        ax_peth.set_xlabel("Time (ms)")
        remove_right_top(ax_peth)

        plt.show()

        break
