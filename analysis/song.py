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


def load_song(database):
    """
    Load information about song path, names
    reads from the song database
    Args:
        database: SQL object (database row)

    Returns:
        song path: path
        song name: str
    """
    song_id = ''
    if len(database['id']) == 1:
        song_id = '00' + database['id']
    elif len(database['id']) == 2:
        song_id = '0' + database['id']
    taskSession = ''
    if len(str(database['taskSession'])) == 1:
        taskSession = 'D0' + str(database['taskSession'])
    elif len(str(database['taskSession'])) == 2:
        taskSession = 'D' + str(database['taskSession'])
    taskSession += '(' + str(database['sessionDate']) + ')'

    song_name = [song_id, database['birdID'], database['taskName'], taskSession]
    song_name = '-'.join(map(str, song_name))

    # Get cluster path
    project_path = ProjectLoader().path
    song_path = project_path / database['birdID'] / database['taskName'] / taskSession
    song_path = Path(song_path)

    return song_name, song_path


class SongInfo:
    def __init__(self, path, format='wav', *name, update=False):

        #Set all database fields as attributes (song database)
        for col in database.keys():
            # dic[col] = database[col]
            setattr(self, col, database[col])

        # Get cluster name & path
        self.name, self.path = load_song(database)
        print('')
        print('Load song {self.name}'.format(self=self))
