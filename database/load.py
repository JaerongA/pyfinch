import sqlite3
import pandas as pd


def database(*query):
    # Apply query to the database
    # Return cursor
    conn = sqlite3.connect('database/deafening.db')
    # conn.row_factory = lambda cursor, row: row[0]
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if query:
        cur = conn.execute(query[0])
    return cur, conn


def load_cluster(conn, cluster_run):

    from types import SimpleNamespace

    df = pd.read_sql_query("SELECT * FROM cluster WHERE id = (?)", conn, params=(cluster_run,))
    this_dic = df.iloc[0].to_dict()
    cell = SimpleNamespace(**this_dic)

    if len(cell.id) == 1:
        cell.id = '00' + cell.id
    elif len(cell.id) == 2:
        cell.id = '0' + cell.id

    if len(str(cell.taskSession)) == 1:
        cell.taskSession = 'D0' + str(cell.taskSession)
    elif len(str(cell.taskSession)) == 2:
        cell.taskSession = 'D' + str(cell.taskSession)
    cell.site = cell.site[-2:]

    cell_name = cell.id + '-' + cell.birdID + '-' + cell.taskName + '-' + cell.taskSession + '-' + cell.sessionDate + '-' + cell.channel + '-' + cell.cluster
    session_path = project_path + '\\' + cluster.BirdID + '\\' + cluster.TaskName + '\\' + cluster.TaskSession + '(' + cluster.SessionDate + ')'
    cell_path = session_path + '\\' + cluster.Site + '\\Songs'

    return cell, cell_name, cell_path


def load_song(conn, cluster_run):
    cur = conn.execute('''SELECT * FROM song''')

    if len(cluster['id']) == 1:
        cluster['id'] = '00' + cluster['id']
    elif len(cluster['id']) == 2:
        cluster['id'] = '0' + cluster['id']

    if len(cluster.TaskSession) == 1:
        cluster.TaskSession = 'D0' + cluster.TaskSession
    elif len(cluster.TaskSession) == 2:
        cluster.TaskSession = 'D' + cluster.TaskSession
    cluster.Site = cluster.Site[-2:]
    return cluster









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


if __name__ == '__main__':
    cur = database()

