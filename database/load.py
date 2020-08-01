import sqlite3
import pandas as pd


def database(*query):
    conn = sqlite3.connect('database/deafening.db')
    # conn = sqlite3.connect('deafening.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if query:
        # cur = conn.execute('''SELECT * FROM cluster''')
        cur = conn.execute(query[0])
    return conn, cur


# for row in rows:
#    print(row)

# cluster_df = pd.read_sql_query('SELECT * FROM cluster', conn)

# conn.row_factory = sqlite3.Row
# cur = conn.execute('''SELECT * FROM cluster''')

# for cluster in cur:
#  if cluster['id'] is 1:
#     print(cluster['birdID'])
#     break

# conn = sqlite3.connect('deafening.db')
# cursor = conn.cursor()
# rows = cursor.execute('''SELECT * FROM cluster''')
# for row in rows:
#    print(row)

cursor = database()
