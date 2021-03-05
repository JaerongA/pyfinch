"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""

from contextlib import suppress

from analysis.spike import *
from database.load import ProjectLoader, DBInfo

# Parameters
update = True
update_db = False

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 1"
# query = "SELECT * FROM cluster"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    motif = cluster_db.motif
    # Load class object
    bi = BaselineInfo(path, channel_nb, unit_nb, format, name, update=update)
    mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name, update=update)  # cluster object


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
