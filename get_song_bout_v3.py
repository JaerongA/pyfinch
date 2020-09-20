"""
By Jaerong
Separates song bouts (*) and store them in the database
"""

from database import load
from pathlib import Path
from song_analysis.parameters import *
import scipy.io
import numpy as np
import sqlite3


def unique_nb_notes_in_bout(note: str, bout: str):
    # returns the unique number of notes within a single bout string
    nb_song_note_in_bout = len([note for note in note if note in bout])
    return nb_song_note_in_bout


def total_nb_notes_in_bout(note: str, bout: str):
    # returns the total number of song notes from a list of song bouts
    notes = []
    nb_notes = []
    for note in note:
        notes.append(note)
        nb_notes.append(sum([bout.count(note) for bout in bout]))

    return sum(nb_notes)


# Load song database

# query = "SELECT * FROM song WHERE id BETWEEN 3 AND 8"
query = "SELECT * FROM song WHERE birdID = 'g35r38'"
# query = "SELECT * FROM song WHERE id =3"

cur, conn, col_names = load.database(query)

for song_info in cur.fetchall():

    song_name, song_path = load.song_info(song_info)
    print(song_name)
    context_list = list()
    bout_list = list()

    for site in [x for x in song_path.iterdir() if x.is_dir()]:  # loop through the sub-dir

        mat_files = [file for file in site.rglob('*.not.mat')]

        for file in mat_files:
            syllables = scipy.io.loadmat(file)['syllables'][0]  # Load the syllable info
            onsets = scipy.io.loadmat(file)['onsets'].transpose()[0]  # syllable onset timestamp
            offsets = scipy.io.loadmat(file)['offsets'].transpose()[0]  # syllable offset timestamp
            intervals = onsets[1:] - offsets[:-1]  # interval among syllables
            context_list.append(
                file.name.split('.')[0].split('_')[-1][0].upper())  # extract 'U' or 'D' from the file name
            print(file)
            # print(file.name)

            # demarcate the song bout with an asterisk (stop)
            ind = np.where(intervals > bout_crit)[0]
            bout_labeling = syllables
            if len(ind):
                for i, item in enumerate(ind):
                    if i is 0:
                        bout_labeling = syllables[:item + 1]
                    else:
                        bout_labeling += '*' + syllables[ind[i - 1] + 1:ind[i] + 1]
                bout_labeling += '*' + syllables[ind[i] + 1:]

            bout_labeling += '*'
            # print(bout_labeling)
            bout_list.append(bout_labeling)

            # count the number of bouts (only includes those having a song note)
            nb_bouts = len(
                [bout for bout in bout_labeling.split('*')[:-1] if unique_nb_notes_in_bout(song_info['songNote'], bout)])

        print(bout_list)

        break

    # Store song bouts and its context in a dict
    bout = {'Undir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'U']),
            'Dir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'D'])
            }

    bout = {'Undir': {'notes': bout['Undir'],
                      'nb_bout': len(
                          [bout for bout in bout['Undir'].split('*')[:-1] if
                           unique_nb_notes_in_bout(song_info['songNote'], bout)])},
            'Dir': {'notes': bout['Dir'],
                    'nb_bout': len(
                        [bout for bout in bout['Dir'].split('*')[:-1] if
                         unique_nb_notes_in_bout(song_info['songNote'], bout)])}
            }

    print(bout)

    # Create new columns
    ## Todo : make this into a function

    if 'songBoutUndir' not in col_names:
        cur.execute("ALTER TABLE song ADD COLUMN songBoutUndir TEXT")

    if 'songBoutDir' not in col_names:
        cur.execute("ALTER TABLE song ADD COLUMN songBoutDir TEXT")

    if 'nbSongBoutUndir' not in col_names:
        cur.execute("ALTER TABLE song ADD COLUMN nbSongBoutUndir INTEGER")

    if 'nbSongBoutDir' not in col_names:
        cur.execute("ALTER TABLE song ADD COLUMN nbSongBoutDir INTEGER")

    # Update the database
    cur.execute("UPDATE song SET songBoutUndir = ? WHERE id = ?", (bout['Undir']['notes'], song_info['id']))
    cur.execute("UPDATE song SET songBoutDir = ? WHERE id = ?", (bout['Dir']['notes'], song_info['id']))

    cur.execute("UPDATE song SET nbSongBoutUndir = ? WHERE id = ?", (bout['Undir']['nb_bout'], song_info['id']))
    cur.execute("UPDATE song SET nbSongBoutDir = ? WHERE id = ?", (bout['Dir']['nb_bout'], song_info['id']))
    conn.commit()
    break
conn.close()

# Load the bout info from the database for analysis

from statistics import mean

# query = "SELECT * FROM song WHERE birdID = 'g35r38'"
query = "SELECT * FROM song WHERE id =3"

cur, conn, col_names = load.database(query)

for song_info in cur.fetchall():

    syllables = song_info['songBoutUndir']
    bout_list = syllables.split('*')[:-1]  # remove bout demarcation and put them in a list

    songbout_list = [bout for bout in bout_list if
                     unique_nb_notes_in_bout(song_info['songNote'], bout)]  # only extracts bouts with song notes

    # Calculate the mean number of introductory notes
    mean_nb_intro_notes = mean(list(map(lambda x: x.count(song_info['introNotes']), songbout_list)))

    # Calculate song/call proportions (# of calls / # of song notes)
    song_call_prop = total_nb_notes_in_bout(song_info['calls'], songbout_list) / total_nb_notes_in_bout(
        song_info['songNote'], songbout_list)

    # Note sequence for the current song bout
    note_seq = list(song_info['introNotes'] + song_info['songNote'] + song_info['calls'] + '*')  # arrange in this order
    bout = song_info['songBoutUndir']
    bout = sorted(set(bout))
    note_seq = [note for note in note_seq if note in bout]  # only count the notes that actually appeared in the bout
    note_seq = ['i', 'a', 'b', 'c', 'd', 'j', 'k', 'm', '*']

    # Construct a transition matrix
    trans_matrix = np.zeros((len(note_seq), len(note_seq)))  # initialize the matrix

    for i, note in enumerate(syllables):

        if i < len(syllables) - 1:
            print(syllables[i] + '->' + syllables[i + 1])
            ind1 = note_seq.index(syllables[i])
            ind2 = note_seq.index(syllables[i + 1])
            trans_matrix[ind1, ind2] = trans_matrix[ind1, ind2] + 1

    # Draw a transition matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.heatmap(trans_matrix, cmap="YlGnBu")
    plt.show()

    break
