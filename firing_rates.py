"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""

from contextlib import suppress

from analysis.spike import *
from database.load import ProjectLoader

# Parameters
update = True
update_db = True

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 2"
# query = "SELECT * FROM cluster"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load class object
    bi = BaselineInfo(row, update=update)
    mi = MotifInfo(row, update=update)

    # Calculate firing rates
    mi.get_mean_fr()

    # Save results to database
    if update_db:
        db.create_col('cluster', 'baselineFR', 'REAL')
        try:
            db.update('cluster', 'baselineFR', row['id'], round(bi.mean_fr, 3))  # baseline firing rates
        except:
            pass
        db.create_col('cluster', 'motifFRUndir', 'REAL')
        try:
            db.update('cluster', 'motifFRUndir', row['id'], round(mi.mean_fr['U'], 3))  # motif firing rates during Undir
        except:
            pass
        db.create_col('cluster', 'motifFRDir', 'REAL')
        try:
            db.update('cluster', 'motifFRDir', row['id'], round(mi.mean_fr['D'], 3))  # motif firing rates during Dir
        except:
            pass

# Convert db to csv
if update_db:
    db.to_csv('cluster')
print('Done!')