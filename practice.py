

from database.load import ProjectLoader, DBInfo

# Load database
db = ProjectLoader().load_db()
# Load database
db = ProjectLoader().load_db()
with open('database/create_unit_profile.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# # Save results
  # # query = "SELECT * FROM unit_pr to database
# # # SQL statement
# # query = "SELECT * FROM cluster WHERE id = 96"
# # # query = "SELECT * FROM cluster WHERE ephysOK=True"
# # db.execute(query)
# #
# # # Loop through db
# # for row in db.cur.fetchall():
# #
# #   cluster_db = DBInfo(row)ofile"
  # # db.execute(query)
  # # db.cur.execute("INSERT OR IGNORE INTO unit_profile(clusterID) VALUES({})".format(cluster_db.id))
  #
  # db.cur.execute("INSERT OR IGNORE INTO unit_profile"
  #                "(clusterID, birdID, taskName, taskSession, site, channel, unit, region) "
  #                "VALUES({}, {}, {}, {}, {}, {}, {}, {})"
  #                .format(cluster_db.id, cluster_db.birdID, cluster_db.taskName, cluster_db.taskSession,
  #                        cluster_db.site, cluster_db.channel, cluster_db.unit, cluster_db.region))
  # db.conn.commit()