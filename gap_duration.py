"""Analyze syllable gap durations"""


# Check if the data .csv exists

from analysis.functions import get_note_type
from analysis.song import SongInfo
from database.load import ProjectLoader, DBInfo
import pandas as pd
import numpy as np
from util import save

# Create save path
save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'GapDuration')

query = "SELECT * FROM song WHERE id=1"

# Load song database
db = ProjectLoader().load_db()
db.execute(query)

# Loop through db
for row in db.cur.fetchall():
    # Load song info from db
    song_db = DBInfo(row)
    name, path = song_db.load_song_db()

    # Load song object
    si = SongInfo(path, name)
    print('\nAccessing... ' + si.name)

    # Store results in the dataframe
    df = pd.DataFrame()

    list_zip = zip(si.onsets, si.offsets, si.syllables, si.contexts)

    for onsets, offsets, syllables, context in list_zip:


        note_types = get_note_type(''.join(si.syllables).replace('*', ''), song_db)  # Type of the syllables

        onsets = onsets[onsets != '*'].astype(np.float)
        offsets = offsets[offsets != '*'].astype(np.float)
        intervals = onsets[1:] - offsets[:-1]
        syllables = syllables.replace('*', '')

        from analysis.functions import find_str
        indices = find_str(syllables, 'a')
        note_interval = []
        for ind in indices:
            note_interval.append(intervals[ind-1])