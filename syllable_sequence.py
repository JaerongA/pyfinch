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


def get_syl_color(bird_id : str):
    """Map colors to each syllable"""
    from analysis.parameters import cmap_list, sequence_color
    from database.load import ProjectLoader, DBInfo

    # Load database
    db = ProjectLoader().load_db()
    # db.execute(f"SELECT introNotes, motif, calls FROM bird WHERE birdID='{bird_id}'")
    df = db.to_dataframe(f"SELECT introNotes, songNote, calls FROM bird WHERE birdID='{bird_id}'")
    intro_notes = df['introNotes'][0]
    song_notes = df['songNote'][0]
    calls = df['calls'][0]

    note_seq = intro_notes + song_notes + calls + '*'

    syl_color = []

    for i, note in enumerate(note_seq[:-1]):
        if note in song_notes:
            syl_color.append(sequence_color['song_note'].pop(0))
        elif note in intro_notes:
            syl_color.append(sequence_color['intro'].pop(0))
        elif note in calls:
            syl_color.append(sequence_color['call'].pop(0))
        else:
            syl_color.append(sequence_color['intro'].pop(0))
    syl_color.append('y')  # for stop at the end
    return syl_color


def get_trans_matrix(syllables, note_seq, norm=False):
    """Build a syllable transition matrix"""
    trans_matrix = np.zeros((len(note_seq), len(note_seq)))  # initialize the matrix
    normalize = 0

    for i, note in enumerate(syllables):

        if i < len(syllables) - 1:
            # print(syllables[i] + '->' + syllables[i + 1])
            ind1 = note_seq.index(syllables[i])
            ind2 = note_seq.index(syllables[i + 1])
            if ind1 < len(note_seq) - 1:
                # trans_matrix[ind1, ind2] = trans_matrix[ind1, ind2] + 1
                trans_matrix[ind1, ind2] += 1

    if norm:
        print("normalize")
        trans_matrix = trans_matrix / trans_matrix.sum()
    return trans_matrix


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

    si = SongInfo(path, name, update=update_cache)  # song object

    # Get syllable color
    syl_color = get_syl_color(song_db.birdID)

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

        trans_matrix = get_trans_matrix(song_bouts, note_seq)
        print(trans_matrix)