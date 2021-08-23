"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
Get firing rates from song motif (including pre-motor windows)
"""

def get_firing_rates(query, update=False, update_db=False):

    from analysis.spike import BaselineInfo, MotifInfo
    from database.load import ProjectLoader, DBInfo
    from analysis.parameters import nb_note_crit

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        unit_nb = int(cluster_db.unit[-2:])
        channel_nb = int(cluster_db.channel[-2:])
        format = cluster_db.format
        motif = cluster_db.motif

        # Load class object
        bi = BaselineInfo(path, channel_nb, unit_nb, format, name, update=update)
        mi = MotifInfo(path, channel_nb, unit_nb, motif, format, name, update=update)  # cluster object

        # Get number of motifs
        nb_motifs = mi.nb_motifs(motif)
        nb_motifs.pop('All', None)

        # Calculate firing rates
        mi.get_mean_fr()

        # Save results to database
        if update_db:
            db.create_col('cluster', 'baselineFR', 'REAL')
            db.update('cluster', 'baselineFR', row['id'], round(bi.mean_fr, 3))  # baseline firing rates

            db.create_col('cluster', 'motifFRUndir', 'REAL')
            if 'U' in mi.mean_fr and nb_motifs['U'] >= nb_note_crit:
                db.update('cluster', 'motifFRUndir', row['id'], round(mi.mean_fr['U'], 3))  # motif firing rates during Undir

            db.create_col('cluster', 'motifFRDir', 'REAL')
            if 'D' in mi.mean_fr and nb_motifs['D'] >= nb_note_crit:
                db.update('cluster', 'motifFRDir', row['id'], round(mi.mean_fr['D'], 3))  # motif firing rates during Dir

    # Convert db to csv
    if update_db:
        db.to_csv('cluster')
    print('Done!')


if __name__ =='__main__':

    # Parameters
    update = False
    update_db = True

    # SQL statement (select from cluster db)
    query = "SELECT * FROM cluster WHERE analysisOK = 1"

    get_firing_rates(query, update=update, update_db=update_db)