# demarcate bout
# extract audio, neural data (by using class)

from analysis.spike import *
from util import save
from util.draw import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from database.load import DBInfo
from scipy import stats

# Parameters
save_fig = True
update = False
dir_name = 'RasterBouts'
fig_ext = '.png'  # .png or .pdf
font_size = 12  # figure font size
rec_yloc = 0.05
rect_height = 0.3
text_yloc = 1  # text height
nb_row = 13
nb_col = 1
tick_length = 1
tick_width = 1

class BoutInfo(ClusterInfo):
    """Child class of ClusterInfo"""

    def __init__(self, path, channel_nb, unit_nb, song_note, format='rhd', *name, update=False):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        self.song_note = song_note

        if name:
           self.name = name[0]
        else:
           self.name = str(self.path)

        file_name = self.path / "BoutInfo_{}_Cluster{}.npy".format(self.channel_nb, self.unit_nb)
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            bout_info = self.load_bouts()
            # Save event_info as a numpy object
            np.save(file_name, bout_info)
        else:
            bout_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in bout_info:
            setattr(self, key, bout_info[key])

    def print_name(self):
        print('')
        print('Load bout {self.name}'.format(self=self))

    def __len__(self):
        return len(self.files)

    def load_bouts(self):
        # Store values here
        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
        context_list = []

        list_zip = zip(self.files, self.spk_ts, self.onsets, self.offsets, self.syllables, self.contexts)

        for file, spks, onsets, offsets, syllables, context in list_zip:

            bout_ind = find_str(syllables, '*')

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

        return bout_info


# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 6"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format

    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    bi = BoutInfo(path, channel_nb, unit_nb, cluster_db.songNote, format, name, update=update)  # bout object

    list_zip = zip(bi.files, bi.spk_ts, bi.onsets, bi.offsets, bi.syllables, bi.contexts)

    for bout_ind, (file, spks, onsets, offsets, syllables, context) in enumerate(list_zip):

        # Convert from string to array of floats
        onsets = np.asarray(list(map(float, onsets)))
        offsets = np.asarray(list(map(float, offsets)))
        spks = spks - onsets[0]

        # bout start and end
        start = onsets[0] - peth_parm['buffer']
        end = offsets[-1] + peth_parm['buffer']
        duration = offsets[-1] - onsets[0]

        # Get spectrogram
        audio = AudioData(path, update=update).extract([start, end])  # audio object
        audio.spectrogram()
        audio.spect_time = audio.spect_time - audio.spect_time[0] - peth_parm['buffer']

        # Plot figure
        fig = plt.figure(figsize=(8, 7))
        fig.tight_layout()
        fig_name = f"{file} - Bout # {bout_ind + 1}"
        fig.suptitle(fig_name, y=0.95)

        # Plot spectrogram
        ax_spect = plt.subplot2grid((nb_row, nb_col), (2, 0), rowspan=2, colspan=1)
        ax_spect.pcolormesh(audio.spect_time, audio.spect_freq, audio.spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
        plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
        plt.setp(ax_spect.get_xticklabels(), visible=False)
        plt.xlim([audio.spect_time[0]-100, audio.spect_time[-1]+100])

        # Plot syllable duration
        ax_syl = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=1, colspan=1, sharex=ax_spect)
        note_dur = offsets - onsets  # syllable duration
        onsets -= onsets[0]  # start from 0
        offsets = onsets + note_dur

        # Mark syllables
        for i, syl in enumerate(syllables):
            rectangle = plt.Rectangle((onsets[i], rec_yloc), note_dur[i], rect_height,
                                      linewidth=1, alpha=0.5, edgecolor='k', facecolor=bout_color[syl])
            ax_syl.add_patch(rectangle)
            ax_syl.text((onsets[i] + (offsets[i] - onsets[i]) / 2), text_yloc, syl, size=font_size)
        ax_syl.axis('off')

        # Plot song amplitude
        audio.data = stats.zscore(audio.data)
        audio.timestamp = audio.timestamp - audio.timestamp[0] - peth_parm['buffer']
        ax_amp = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=2, colspan=1, sharex=ax_spect)
        ax_amp.plot(audio.timestamp, audio.data, 'k', lw=0.1)
        ax_amp.axis('off')

        # Plot rasters
        ax_raster = plt.subplot2grid((nb_row, nb_col), (6, 0), rowspan=2, colspan=1, sharex=ax_spect)
        # spks2 = spks - start -peth_parm['buffer'] -peth_parm['buffer']
        ax_raster.eventplot(spks, colors='k', lineoffsets=0.5,
                            linelengths=tick_length, linewidths=tick_width, orientation='horizontal')
        ax_raster.axis('off')

        # Plot raw neural data
        nd = NeuralData(path, channel_nb, format, update=update).extract([start, end])  # raw neural data
        nd.timestamp = nd.timestamp - nd.timestamp[0] - peth_parm['buffer']
        ax_nd = plt.subplot2grid((nb_row, nb_col), (8, 0), rowspan=2, colspan=1, sharex=ax_spect)
        ax_nd.plot(nd.timestamp, nd.data, 'k', lw=0.5)

        # Add a scale bar
        plt.plot([ax_nd.get_xlim()[0] + 50, ax_nd.get_xlim()[0] + 50],
                 [-250, 250], 'k', lw=3)  # for amplitude
        plt.text(ax_nd.get_xlim()[0] , -200, '500 µV', rotation=90)
        plt.subplots_adjust(wspace=0, hspace=0)
        remove_right_top(ax_nd)
        ax_nd.spines['left'].set_visible(False)
        plt.yticks([], [])
        ax_nd.set_xlabel('Time (ms)')




        plt.show()

        break
