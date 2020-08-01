import sqlite3
import pandas as pd


def database():
    conn = sqlite3.connect('deafening.db')
    cursor = conn.cursor()
    rows = cursor.execute('''SELECT * FROM cluster''')
    for row in rows:
        print(row)

    #conn.row_factory = sqlite3.Row
    #cursor = conn.execute('''SELECT * FROM cluster''')
    return cursor
    # rows = cur.execute('''SELECT * FROM cluster''')

# for row in rows:
#    print(row)

# cluster_df = pd.read_sql_query('SELECT * FROM cluster', conn)

# conn.row_factory = sqlite3.Row
# cur = conn.execute('''SELECT * FROM cluster''')

# for cluster in cur:
#  if cluster['id'] is 1:
#     print(cluster['birdID'])
#     break

#conn = sqlite3.connect('deafening.db')
#cursor = conn.cursor()
#rows = cursor.execute('''SELECT * FROM cluster''')
#for row in rows:
#    print(row)

cursor = database()