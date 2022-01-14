"""
Analyze syllable gap durations
"""


def main(query,
         update=False,
         update_db=False,
         ):
    from analysis.functions import find_str
    from analysis.song import SongInfo
    from collections import defaultdict
    from database.load import ProjectLoader, DBInfo, create_db
    from functools import partial
    import numpy as np

    # Create & Load database
    if update_db:
        db = create_db('create_gap_duration.sql')

    # Load song database
    db = ProjectLoader().load_db()
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        # Load song object
        si = SongInfo(path, name, update=update)
        print('\nAccessing... ' + si.name)

        list_zip = zip(si.onsets, si.offsets, si.syllables, si.contexts)

        pre_syllable_gaps = defaultdict(partial(np.ndarray, 0))

        for onsets, offsets, syllables, context in list_zip:

            onsets = onsets[onsets != '*'].astype(np.float)
            offsets = offsets[offsets != '*'].astype(np.float)
            syllables = syllables.replace('*', '')
            gaps = onsets[1:] - offsets[:-1]  # gap durations of all syllables

            # Loop over notes
            for note in song_db.songNote:
                note_indices = find_str(syllables, note)

                for ind in note_indices:
                    # Get pre-motor gap duration for each song syllable
                    pre_syllable_gap = gaps[ind - 1]
                    pre_syllable_gaps[note] = np.append(pre_syllable_gaps[note], pre_syllable_gap)

                if update_db:
                    sql = f"""INSERT INTO gap_duration (songID, birdID, taskName, note, context, gap_duration)
                          VALUES ('song_db.id', 'song_db.birdID', 'song_db.taskName', 'note', 'context', 'pre_syllable_gap')"""
                    db.cur.execute(sql)

    # Convert db to csv
    if update_db:
        db.to_csv('gap_duration')
    print('Done!')


if __name__ == '__main__':

    # Parameters
    update = False  # create/update .json cache file
    update_db = True  # create/update database

    # SQL statement
    query = "SELECT * FROM song"

    main(query,
         update=update,
         update_db=update_db,
         )
