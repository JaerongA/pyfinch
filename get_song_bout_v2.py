"""
By Jaerong
Separates song bouts (*) and store them in .json in song folders
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


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


query = "SELECT * FROM song WHERE id BETWEEN 2 AND 3"  # Load song database

cur, conn = load.database(query)

nb_rows = len(cur.fetchall())
print(nb_rows)
for song_run in range(0, nb_rows):

    song, song_name, song_path = load.song_info(conn, song_run)



    print(song_name)

