"""
A package for song analysis
"""

from analysis.functions import *
# from analysis.parameters import *
from analysis.load import *
from database.load import ProjectLoader
from pathlib import Path
from util.functions import *
from util.spect import *


# def load_song(database):
#     """
#     Load information about song path, names
#     reads from the song database
#     Args:
#         database: SQL object (database row)
#
#     Returns:
#         song path: path
#         song name: str
#     """
#     song_id = ''
#     if len(database['id']) == 1:
#         song_id = '00' + database['id']
#     elif len(database['id']) == 2:
#         song_id = '0' + database['id']
#     taskSession = ''
#     if len(str(database['taskSession'])) == 1:
#         taskSession = 'D0' + str(database['taskSession'])
#     elif len(str(database['taskSession'])) == 2:
#         taskSession = 'D' + str(database['taskSession'])
#     taskSession += '(' + str(database['sessionDate']) + ')'
#
#     song_name = [song_id, database['birdID'], database['taskName'], taskSession]
#     song_name = '-'.join(map(str, song_name))
#
#     # Get cluster path
#     project_path = ProjectLoader().path
#     song_path = project_path / database['birdID'] / database['taskName'] / taskSession
#     song_path = Path(song_path)
#
#     return song_name, song_path


def load_song(dir, format='wav'):
    """
    Obtain event info & serialized timestamps for song & neural analysis
    """
    import numpy as np
    from scipy.io import wavfile

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

    def __init__(self, path, format='wav', *name, update=False):

        self.path = path
        self.format = format
        if name:
            self.name = name[0]
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

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def print_name(self):
        print('')
        print('Load song info {self.name}'.format(self=self))

    def list_files(self, ext: str):
        return list_files(self.path, ext)