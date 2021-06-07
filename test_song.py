"""
By Jaerong
Song analysis
"""

import matplotlib.pyplot as plt

from analysis.parameters import *
from analysis.song import *
from database.load import ProjectLoader, DBInfo
from util import save


# Parameter
nb_row = 8
nb_col = 3
normalize = False  # normalize correlogram
update = False
save_fig = False
update_db = False  # save results to DB
fig_ext = '.png'  # .png or .pdf

# Load database
db = ProjectLoader().load_db()
with open('database/create_song_table.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# SQL statement
query = "SELECT * FROM song WHERE id=3"
# query = "SELECT * FROM cluster WHERE ephysOK=True"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():
    # Load song info from db
    song_db = DBInfo(row)
    name, path = song_db.load_song_db()

    si = SongInfo(path, name, update=update)  # cluster object

    nb_files = si.nb_files
    nb_bouts = si.nb_bouts(song_db.songNote)
    nb_motifs = si.nb_motifs(song_db.motif)

    mean_nb_intro_notes = si.mean_nb_intro(song_db.introNotes, song_db.songNote)
    song_call_prop = si.song_call_prop(song_db.introNotes, song_db.songNote)
    mi = si.get_motif_info(song_db.motif)  # Get motif info
    motif_dur = mi.get_motif_duration()  # Get mean motif duration &  CV per context

    if update_db:
        db.cur.execute("UPDATE song SET nbFilesUndir=?, nbFilesDir=? WHERE id=?", (nb_files['U'], nb_files['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbBoutsUndir=?, nbBoutsDir=? WHERE id=?", (nb_bouts['U'], nb_bouts['D'], song_db.id))
        db.cur.execute("UPDATE song SET nbMotifsUndir=?, nbMotifsDir=? WHERE id=?", (nb_motifs['U'], nb_motifs['D'], song_db.id))
        db.cur.execute("UPDATE song SET meanIntroUndir=?, meanIntroDir=? WHERE id=?", (mean_nb_intro_notes['U'], mean_nb_intro_notes['D'], song_db.id))
        db.cur.execute("UPDATE song SET songCallPropUndir=?, songCallPropDir=? WHERE id=?", (song_call_prop['U'], song_call_prop['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationUndir=?, motifDurationDir=? WHERE id=?", (motif_dur['mean']['U'], motif_dur['mean']['D'], song_db.id))
        db.cur.execute("UPDATE song SET motifDurationCVUndir=?, motifDurationCVDir=? WHERE id=?", (motif_dur['cv']['U'], motif_dur['cv']['D'], song_db.id))

print('Done!')