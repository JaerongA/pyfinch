import sqlite3
import pandas as pd

conn = sqlite3.connect('deafening.db')

cur = conn.cursor()

cur.execute('''SELECT * FROM cluster''')


cluster_df = pd.read_sql_query('SELECT * FROM cluster', conn)

