"""
By Jaerong
A package for song analysis
"""

# from analysis.functions import *


# # from analysis.parameters import *
# from analysis.load import *
# from database.load import ProjectLoader
# from pathlib import Path
# from util.spect import *


def load_song(dir, format='wav'):
    """
    Obtain event info & serialized timestamps for song & neural analysis
    """
    from analysis.functions import  demarcate_bout, read_not_mat
    import numpy as np
    from scipy.io import wavfile
    from util.functions import list_files

    # List all audio files in the dir
    audio_files = list_files(dir, format)

    # Initialize
    timestamp_serialized = np.array([], dtype=np.float32)

    # Store values in these lists
    file_list = []
    file_start_list = []
    file_end_list = []
    onset_list = []
    offset_list = []
    duration_list = []
    syllable_list = []
    context_list = []

    # Loop through Intan .rhd files
    for file in audio_files:

        # Load audio files
        print('Loading... ' + file.stem)
        sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
        length = data.shape[0] / sample_rate
        timestamp = np.linspace(0., length, data.shape[0]) * 1E3  # start from t = 0 in ms

        # Load the .not.mat file
        notmat_file = file.with_suffix('.wav.not.mat')
        onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')
        start_ind = timestamp_serialized.size  # start of the file

        if timestamp_serialized.size:
            timestamp += (timestamp_serialized[-1] + (1 / sample_rate))
        timestamp_serialized = np.append(timestamp_serialized, timestamp)

        # File information (name, start & end timestamp of each file)
        file_list.append(file.stem)
        file_start_list.append(timestamp_serialized[start_ind])  # in ms
        file_end_list.append(timestamp_serialized[-1])  # in ms

        onsets += timestamp[0]
        offsets += timestamp[0]

        # Demarcate song bouts
        onset_list.append(demarcate_bout(onsets, intervals))
        offset_list.append(demarcate_bout(offsets, intervals))
        duration_list.append(demarcate_bout(durations, intervals))
        syllable_list.append(demarcate_bout(syllables, intervals))
        context_list.append(contexts)

    # Organize event-related info into a single dictionary object
    song_info = {
        'files': file_list,
        'file_start': file_start_list,
        'file_end': file_end_list,
        'onsets': onset_list,
        'offsets': offset_list,
        'durations': duration_list,
        'syllables': syllable_list,
        'contexts': context_list
    }
    return song_info


class SongInfo:

    def __init__(self, path, name=None, update=False):

        import numpy as np
        from util.functions import list_files

        self.path = path
        if name:
            self.name = name
        else:
            self.name = self.path
        self.print_name()

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

    def print_name(self):
        print('')
        print('Load song info {self.name}'.format(self=self))

    def list_files(self, ext='wav'):
        from util.functions import list_files
        return list_files(self.path, ext)

    def __len__(self):
        return len(self.files)

    @property
    def open_folder(self):
        from util.functions import open_folder
        open_folder(self.path)

    @property
    def nb_files(self):

        nb_files = {}
        nb_files['U'] = len([context for context in self.contexts if context == 'U'])
        nb_files['D'] = len([context for context in self.contexts if context == 'D'])
        nb_files['All'] = nb_files['U'] + nb_files['D']

        return nb_files

    def nb_bouts(self, song_note):

        from analysis.functions import get_nb_bouts

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

        from analysis.functions import find_str

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
        """Mean number of intro notes per song bout
        Only counts from bouts having at least one song note
        """
        from analysis.functions import unique_nb_notes_in_bout
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
            mean_nb_intro_notes[context1] = round(mean(list(map(lambda x: x.count(intro_note), bout_list))), 3)
        return mean_nb_intro_notes

    def song_call_prop(self, call: str, song_note: str):
        """Calculate the proportion of call notes per song bout
        Get the proportion per bout and then average
        only counts from bouts having at least one song note"""

        from analysis.functions import unique_nb_notes_in_bout, total_nb_notes_in_bout
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

            nb_calls_per_bout = np.array(list(map(lambda x: total_nb_notes_in_bout(call, x), bout_list)))
            total_nb_notes = np.array([len(bout) for bout in bout_list])
            song_call_prop[context1] = round((nb_calls_per_bout / total_nb_notes).mean())

        return song_call_prop

    def get_motif_info(self, motif: str):
        """Get information about song motif for the songs recorded in the session"""

        from analysis.functions import find_str

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
        from statistics import mean

        import numpy as np

        motif_dur = {'mean' : {'U': None, 'D': None},
                     'cv' : {'U': None, 'D': None}}

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
