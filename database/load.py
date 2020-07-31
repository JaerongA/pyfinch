import sqlite3
import pandas as pd

conn = sqlite3.connect('deafening.db')

cur = conn.cursor()

#rows = cur.execute('''SELECT * FROM cluster''')

#for row in rows:
#    print(row)

#cluster_df = pd.read_sql_query('SELECT * FROM cluster', conn)

conn.row_factory = sqlite3.Row
cur = conn.execute('''SELECT * FROM cluster''')

for cluster in cur:
    if cluster['id'] is 1:
        print(cluster['birdID'])
        break
