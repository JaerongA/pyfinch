"""
Syllable sequence analysis and calculates transition entropy
"""

from analysis.song import SongInfo
from analysis.parameters import *
from database.load import ProjectLoader, DBInfo
import os
import scipy.io
import numpy as np
from song.functions import *

def nb_song_note_in_bout(song_notes, bout):
    # returns the number of song notes within a bout
    nb_song_note_in_bout = len([note for note in song_notes if note in bout])
    return nb_song_note_in_bout


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


# Parameters
nb_row = 3
nb_col = 4
update_cache = False

# Load database
db = ProjectLoader().load_db()
# # SQL statement
query = "SELECT * FROM song WHERE id = 79"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():
    # Load song info from db
    song_db = DBInfo(row)
    name, path = song_db.load_song_db()

    db.execute(f"SELECT introNotes, motif, calls FROM bird WHERE birdID='{song_db.birdID}'")
    song_seq = [data[0] + data[1] + data[2] for data in db.cur.fetchall()][0]  # e.g., ('iabcdelmn')

    si = SongInfo(path, name, update=update_cache)  # song object

    # Get song bout strings and number of bouts per context
    song_bouts = dict()
    nb_bouts = dict()  # nb of song bouts per context

    for context in sorted(set(si.contexts), reverse=True):
        bout_list = []
        syllable_list = [syllable for syllable, _context in zip(si.syllables, si.contexts) if _context == context]
        for syllables in syllable_list:
            bout = [bout for bout in syllables.split('*') if nb_song_note_in_bout(song_db.songNote, bout)]
            if bout:
                bout_list.append(bout[0])
        song_bouts[context] = '*'.join(bout_list) + '*'
        nb_bouts[context] = len(song_bouts[context].split('*')[:-1])

