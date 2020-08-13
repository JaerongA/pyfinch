"""
By Jaerong
Separates song bouts (*) and store them in the database
"""

import os
from database import load
from pathlib import Path
from song_analysis.parameters import *
import scipy.io
import numpy as np
import sqlite3


def is_song_bout(song_notes, bout):
    # returns the number of song notes within a bout
    nb_song_note_in_bout = len([note for note in song_notes if note in bout])
    return nb_song_note_in_bout


# Load song database
query = "SELECT * FROM song WHERE id BETWEEN 3 AND 4"
# query = "SELECT * FROM song WHERE birdID = 'g35r38'"  # Load song database

cur, conn = load.database(query)

for song_row in cur.fetchall():

    song_name, song_path = load.song_info(song_row)

    context_list = list()
    bout_list = list()

    for site in [x for x in song_path.iterdir() if x.is_dir()]: # loop through the sub-dir

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
            nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if is_song_bout(song_row['songNote'], bout)])

        print(bout_list)

    # Store song bouts and its context in a dict
    bout = {'Undir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'U']),
            'Dir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'D'])
            }

    bout = {'Undir': {'notes': bout['Undir'],
                      'nb_bout': len(
                          [bout for bout in bout['Undir'].split('*')[:-1] if
                           is_song_bout(song_row['songNote'], bout)])},
            'Dir': {'notes': bout['Dir'],
                    'nb_bout': len(
                        [bout for bout in bout['Dir'].split('*')[:-1] if is_song_bout(song_row['songNote'], bout)])}
            }

    print(bout)

    break
