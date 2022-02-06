"""
Performs basic analysis on the song such as mean # of intro notes, song call proportion, motif duration, etc
read info from the song table
stores the results in the song table
"""

def analyze_song(query, update_cache=False, update_db=True):
    """
    Performs song analysis
    Parameters
    ----------
    query : str
        query the song table
    update_cache : bool
        update or create .json file in the data directory
    update_db : bool
        update the song table
    """
    from analysis.song import SongInfo
    from database.load import ProjectLoader, DBInfo

    # Load database
    db = ProjectLoader().load_db()
    with open('../database/create_song_table.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        si = SongInfo(path, name, update=update_cache)  # song object

        nb_files = si.nb_files
        nb_bouts = si.nb_bouts(song_db.songNote)
        nb_motifs = si.nb_motifs(song_db.motif)

        mean_nb_intro_notes = si.mean_nb_intro(song_db.introNotes, song_db.songNote)
        song_call_prop = si.song_call_prop(song_db.calls, song_db.songNote)
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
            db.conn.commit()
        else:
            print(nb_files, nb_bouts, nb_motifs, mean_nb_intro_notes, song_call_prop, motif_dur)

    if update_db:
        db.to_csv(f'song')
    print('Done!')


if __name__ == "__main__":

    # Parameters
    fig_ext = '.png'
    save_fig = False
    update_db = True
    update_cache = False

    # SQL statement
    query = "SELECT * FROM song"
    analyze_song(query, update_cache=update_cache, update_db=update_db )
