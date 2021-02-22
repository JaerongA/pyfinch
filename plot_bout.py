# demarcate bout
# extract audio, neural data (by using class)

from analysis.spike import *
from util import save
from util.draw import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from database.load import DBInfo

# Parameters
save_fig = True
update = False
dir_name = 'RasterBouts'
fig_ext = '.png'  # .png or .pdf


class BoutInfo(ClusterInfo):
    """Child class of ClusterInfo"""

    def __init__(self, path, channel_nb, unit_nb, song_note, format='rhd', update=False):
        self.path = path
        self.song_note = song_note

        file_name = self.path / 'BoutInfo.npy'









# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 6"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format

    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    nd = NeuralData(path, channel_nb, format, update=update)  # raw neural data
    audio = AudioData(path, update=update)  # audio object

    ci.nb_bouts(cluster_db.songNote)

    # Store values here
    file_list = []
    spk_list = []
    onset_list = []
    offset_list = []
    syllable_list = []
    duration_list = []
    context_list = []

    list_zip = zip(ci.files, ci.spk_ts, ci.onsets, ci.offsets, ci.syllables, ci.contexts)

    for file, spks, onsets, offsets, syllables, context in list_zip:

        bout_ind = find_str(syllables, '*')
        # bout_ind.insert(0, 0)  # start from zero

        for ind in range(len(bout_ind)):
            if ind == 0:
                start_ind = 0
            else:
                start_ind = bout_ind[ind - 1] + 1
            stop_ind = bout_ind[ind] - 1
            # breakpoint()
            bout_onset = float(onsets[start_ind])
            bout_offset = float(offsets[stop_ind])

            bout_spk = spks[np.where((spks >= bout_onset) & (spks <= bout_offset))]
            onsets_in_bout = onsets[start_ind:stop_ind + 1]  # list of bout onset timestamps
            offsets_in_bout = offsets[start_ind:stop_ind + 1]  # list of bout offset timestamps

            file_list.append(file)
            spk_list.append(bout_spk)
            duration_list.append(bout_offset - bout_onset)
            onset_list.append(onsets_in_bout)
            offset_list.append(offsets_in_bout)
            syllable_list.append(syllables[start_ind:stop_ind + 1])
            context_list.append(context)

    # Organize event-related info into a single dictionary object
    bout_info = {
        'files': file_list,
        'spk_ts': spk_list,
        'onsets': onset_list,
        'offsets': offset_list,
        'durations': duration_list,  # this is bout durations
        'syllables': syllable_list,
        'contexts': context_list,
    }

    # Bout start and end
    start = onset[0] - peth_parm['buffer']
    end = offset[-1] + peth_parm['buffer']
    duration = offset[-1] - onset[0]

    # Get spectrogram
    audio = AudioData(row).extract([start, end])
    audio.spectrogram(freq_range=freq_range)

    # Plot figure
    fig = plt.figure(figsize=(8, 9), dpi=800)
    # Plot spectrogram
    ax_spect = plt.subplot(gs[1:3, 0:4])
    ax_spect.pcolormesh(audio.timebins * 1E3 - peth_parm['buffer'], audio.freqbins, audio.spect,  # data
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
