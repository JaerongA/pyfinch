"""
Module for song analysis
"""

import numpy as np


class SongInfo:

    def __init__(self, path, name=None, update=False):

        from ..analysis.load import load_song

        self.path = path
        if name:
            self.name = name
        else:
            self.name = self.path
        self.__print_name()

        # Load song
        file_name = self.path / "SongInfo.npy"
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            song_info = load_song(self.path)
            # Save song_info as a numpy object
            np.save(file_name, song_info)
        else:
            song_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in song_info:
            setattr(self, key, song_info[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def __print_name(self):
        print('')
        print('Load song info {self.name}'.format(self=self))

    def list_files(self, ext='.wav'):
        from ..utils.functions import list_files
        return list_files(self.path, ext)

    def __len__(self):
        return len(self.files)

    @property
    def open_folder(self):

        from ..utils.functions import open_folder

        open_folder(self.path)

    @property
    def nb_files(self) -> int:
        """Number of files"""
        nb_files = {}
        nb_files['U'] = len([context for context in self.contexts if context == 'U'])
        nb_files['D'] = len([context for context in self.contexts if context == 'D'])
        nb_files['All'] = nb_files['U'] + nb_files['D']

        return nb_files

    def nb_bouts(self, song_note: str):
        """
        Return the number of bouts

        Parameters
        ----------
        song_note : str
            song notes (e.g., 'abcd')

        Returns
        -------
        nb_bouts : dict
        """
        from ..analysis.functions import get_nb_bouts

        nb_bouts = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_bouts['U'] = get_nb_bouts(song_note, syllables)

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_bouts['D'] = get_nb_bouts(song_note, syllables)
        nb_bouts['All'] = nb_bouts['U'] + nb_bouts['D']

        return nb_bouts

    def nb_motifs(self, motif):

        from ..analysis.functions import find_str

        nb_motifs = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_motifs['U'] = len(find_str(syllables, motif))

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_motifs['D'] = len(find_str(syllables, motif))
        nb_motifs['All'] = nb_motifs['U'] + nb_motifs['D']
        return nb_motifs

    def mean_nb_intro(self, intro_note, song_note):
        """
        Return the mean number of intro notes per song bout
        only counts from bouts having at least one song note
        """
        from ..analysis.functions import unique_nb_notes_in_bout
        from statistics import mean

        mean_nb_intro_notes = {}
        mean_nb_intro_notes['U'] = mean_nb_intro_notes['D'] = None

        for context1 in set(self.contexts):
            syllable_list = [syllable for syllable, context2 in zip(self.syllables, self.contexts) if
                             context2 == context1]
            syllables = ''.join(syllable_list)
            bout_list = syllables.split('*')[:-1]  # eliminate demarcate marker
            bout_list = [bout for bout in bout_list if
                         unique_nb_notes_in_bout(song_note, bout)]  # eliminate bouts having no song note
            if bout_list:
                mean_nb_intro_notes[context1] = round(mean(list(map(lambda x: x.count(intro_note), bout_list))), 3)
        return mean_nb_intro_notes

    def song_call_prop(self, call: str, song_note: str):
        """
        Calculate the proportion of call notes per song bout
        Get the proportion per bout and then average
        only counts from bouts having at least one song note
        """

        from ..analysis.functions import unique_nb_notes_in_bout, total_nb_notes_in_bout
        import numpy as np

        song_call_prop = {}
        song_call_prop['U'] = song_call_prop['D'] = None

        for context1 in set(self.contexts):
            syllable_list = [syllable for syllable, context2 in zip(self.syllables, self.contexts) if
                             context2 == context1]
            syllables = ''.join(syllable_list)
            bout_list = syllables.split('*')[:-1]  # eliminate demarcate marker
            bout_list = [bout for bout in bout_list if
                         unique_nb_notes_in_bout(song_note, bout)]  # eliminate bouts having no song note
            if bout_list:
                nb_calls_per_bout = np.array(list(map(lambda x: total_nb_notes_in_bout(call, x), bout_list)))
                total_nb_notes = np.array([len(bout) for bout in bout_list])
                song_call_prop[context1] = round((nb_calls_per_bout / total_nb_notes).mean(), 4)

        return song_call_prop

    def get_motif_info(self, motif: str):
        """Get information about song motif for the songs recorded in the session"""

        from ..analysis.functions import find_str

        # Store values here
        file_list = []
        onset_list = []
        offset_list = []
        duration_list = []
        context_list = []

        list_zip = zip(self.files, self.onsets, self.offsets, self.syllables, self.contexts)

        for file, onsets, offsets, syllables, context in list_zip:
            onsets = onsets.tolist()
            offsets = offsets.tolist()

            # Find motifs
            motif_ind = find_str(syllables, motif)

            # Get syllable, analysis time stamps
            for ind in motif_ind:
                # start (first syllable) and stop (last syllable) index of a motif
                start_ind = ind
                stop_ind = ind + len(motif) - 1

                motif_onset = float(onsets[start_ind])
                motif_offset = float(offsets[stop_ind])

                onsets_in_motif = onsets[start_ind:stop_ind + 1]  # list of motif onset timestamps
                offsets_in_motif = offsets[start_ind:stop_ind + 1]  # list of motif offset timestamps

                file_list.append(file)
                duration_list.append(motif_offset - motif_onset)
                onset_list.append(onsets_in_motif)
                offset_list.append(offsets_in_motif)
                context_list.append(context)

        # Organize event-related info into a single dictionary object
        motif_info = {
            'files': file_list,
            'onsets': onset_list,
            'offsets': offset_list,
            'durations': duration_list,  # this is motif durations
            'contexts': context_list,
        }

        return MotifInfo(motif_info, motif)


class BoutInfo(SongInfo):
    """
    Get song & spike information for a song bout
    Child class of SongInfo
    """

    def __init__(self, path, song_note, name=None, update=False):
        super().__init__(path, song_note, name, update=False)

        import numpy as np

        self.song_note = song_note

        if name:
            self.name = name[0]
        else:
            self.name = str(self.path)

        # Load bout info
        file_name = self.path / "BoutInfo.npy"
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            bout_info = self._load_bouts()
            # Save info dict as a numpy object
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

    def _load_bouts(self):
        # Store values here
        from ..utils.functions import find_str

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

                bout_onset = float(onsets[start_ind])
                bout_offset = float(offsets[stop_ind])

                onsets_in_bout = onsets[start_ind:stop_ind + 1]  # list of bout onset timestamps
                offsets_in_bout = offsets[start_ind:stop_ind + 1]  # list of bout offset timestamps

                file_list.append(file)
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


class MotifInfo:
    """Child class of SongInfo"""

    # def __init__(self, path, motif=None, name=None, update=False):

    def __init__(self, motif_info, motif):

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])

        self.motif = motif

    def get_motif_duration(self):
        """Get mean motif duration and its cv per context"""

        import numpy as np

        motif_dur = {'mean': {'U': None, 'D': None},
                     'cv': {'U': None, 'D': None}}

        for context1 in set(self.contexts):
            duration = np.array([duration for context2, duration in zip(self.contexts, self.durations)
                                 if context2 == context1])
            motif_dur['mean'][context1] = round(duration.mean(), 3)
            motif_dur['cv'][context1] = round(duration.std() / duration.mean(), 3)
        return motif_dur

    def __len__(self):
        return len(self.files)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class AudioInfo:
    """
    Create an audio object from a single audio file (e.g., .wav)
    """

    def __init__(self, filepath, format='.wav'):
        import numpy as np
        from scipy.io import wavfile

        self.path = filepath  # path object
        self.name = filepath.stem
        self.dir = filepath.parent
        self.format = format
        self.sample_rate, self.data = wavfile.read(filepath)  # note that the timestamp is in second
        length = self.data.shape[0] / self.sample_rate
        self.timestamp = np.linspace(0., length, self.data.shape[0]) * 1E3  # start from t = 0 in ms

    def load_notmat(self):
        """Load the .not.mat file"""
        from ..analysis.load import read_not_mat

        notmat_file = self.path.with_suffix('.wav.not.mat')
        self.onsets, self.offsets, self.intervals, self.durations, self.syllables, self.context \
            = read_not_mat(notmat_file, unit='ms')

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    @property
    def open_folder(self):
        from ..utils.functions import open_folder
        open_folder(self.dir)

    def extract(self, time_range):
        """
        Extracts data from the specified range

        Parameters
        ----------
        time_range : list
            list of time stamps [start, end]

        Returns
        -------
        timestamp : np.ndarray
        data : np.ndarray
        """
        import numpy as np

        start = time_range[0]
        end = time_range[-1]

        ind = np.where((self.timestamp >= start) & (self.timestamp <= end))
        return self.timestamp[ind], self.data[ind]

    def spectrogram(self, timestamp, data, freq_range=[300, 8000]):
        """Calculate spectrogram"""
        from ..utils.spect import spectrogram

        spect, spect_freq, _ = spectrogram(data, self.sample_rate, freq_range=freq_range)
        spect_time = np.linspace(timestamp[0], timestamp[-1], spect.shape[1])  # timestamp for spectrogram
        return spect_time, spect, spect_freq

    def get_spectral_entropy(self, spect, normalize=True, mode=None):
        """
        Calculate spectral entropy

        Parameters
        ----------
        normalize : bool
            Get normalized spectral entropy
        mode : str
            Select one from the following {'spectral', ''spectro_temporal'}
        Returns
        -------
        array of spectral entropy
        """
        from ..analysis.functions import get_spectral_entropy

        return get_spectral_entropy(spect, normalize=normalize, mode=mode)

class FundamentalFreq:
    """Class object for analyzing fundamental frequency of a syllable"""
    def __init__self(self, note=None,
                     crit=None, parameter=None, onset=None, offset=None,
                     low=None, high=None, harmonic=1
                     ):

        self.note = note
        self.crit = crit
        self.parameter = None  # {'percent_from_start', 'ms_from_start', 'ms_from_end'}
        self.onset = None
        self.offset = None
        self.low = None
        self.high = None
        self.harmonic = None
        self.value = None  # Fundamental Frequency (FF) value

    def load_from_db(self, birdID, ff_note):
        """Load info from the database if exists"""
        from pyfinch.database.load import ProjectLoader
        query = f"SELECT ffNote, ffParameter, ffCriterion, ffLow, ffHigh, ffDuration, harmonic " \
                f"FROM ff " \
                f"WHERE birdID='{birdID}' AND ffNote='{ff_note}'"
        db = ProjectLoader().load_db().execute(query)

        ff_info = {data[0]: {'parameter': data[1],
                             'crit': data[2],
                             'low': data[3],  # lower limit of frequency
                             'high': data[4],  # upper limit of frequency
                             'duration': data[5],
                             'harmonic': data[6]  # 1st or 2nd harmonic detection
                             } for data in db.cur.fetchall()  # ff duration
                   }

        # Set the dictionary values to class attributes
        for key in ff_info:
            setattr(self, key, ff_info[key])

    def get_ts(self, note_onset, note_offset):
        """Get onset and offset timestamp of FF portion based on note onset & offset"""
        pass

class SyllableNetwork:

    def __init__(self):
        pass

