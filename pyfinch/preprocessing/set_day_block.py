"""Compare motif firing rates between different conditions"""

from pyfinch.database.load import ProjectLoader

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster")
db.cur.execute(f"ALTER TABLE {cluster} ADD COLUMN {col_name} {INT}")
db.cur.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, col_name, type))
