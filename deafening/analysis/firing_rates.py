"""
Run firing rate analysis
Get mean firing rates per condition
Get firing rates from song motif (including pre-motor windows)
Stores the results in the unit_profile table
"""
from analysis.parameters import nb_note_crit
from analysis.spike import BaselineInfo, MotifInfo
from database.load import ProjectLoader, DBInfo, create_db

def get_firing_rates():

    # Create & Load database
    if update_db:
        create_db('create_unit_profile.sql')

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
        mi.get_mean_fr(add_pre_motor=add_pre_motor)

        # Save results to database
        if update_db:
            db.cur.execute(f"""UPDATE unit_profile SET baselineFR = ({bi.mean_fr}) WHERE clusterID = {cluster_db.id}""")
            if 'U' in mi.mean_fr and nb_motifs['U'] >= nb_note_crit:
                db.cur.execute(f"""UPDATE unit_profile SET motifFRUndir = ({mi.mean_fr['U']}) WHERE clusterID = {cluster_db.id}""")
            if 'D' in mi.mean_fr and nb_motifs['D'] >= nb_note_crit:
                db.cur.execute(f"""UPDATE unit_profile SET motifFRDir = ({mi.mean_fr['D']}) WHERE clusterID = {cluster_db.id}""")
            db.cur.execute(query)
            db.conn.commit()

    # Convert db to csv
    if update_db:
        db.to_csv('unit_profile')
    print('Done!')


if __name__ == '__main__':

    # Parameters
    update = False
    update_db = False
    add_pre_motor = True  # add spikes from the pre-motor window

    # SQL statement (select from cluster db)
    query = "SELECT * FROM cluster WHERE analysisOK"

    get_firing_rates()
