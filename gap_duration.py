"""Analyze syllable gap durations"""


# Check if the data .csv exists
from analysis.song import SongInfo
from database.load import ProjectLoader, DBInfo
import pandas as pd
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



