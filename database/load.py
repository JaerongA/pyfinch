import sqlite3

conn = sqlite3.connect('deafening.db')

cur = conn.cursor()


cur.execute('SELECT * FROM cluster')
