"""
A package for song analysis
"""

from analysis.functions import *
from analysis.parameters import *
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


# def load_song(dir):
#
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