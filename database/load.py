import sqlite3
import pandas as pd
import os


def config():
    from configparser import ConfigParser
    config_file = 'database/project.ini'
    parser = ConfigParser()
    parser.read(config_file)
    print(parser.sections())
    return parser


def project():
    from configparser import ConfigParser
    config_file = 'database/project.ini'
    parser = ConfigParser()
    parser.read(config_file)
    project_path = parser.get('folders', 'project_path')
    return project_path


def database(*query):
    # Apply query to the database
    # Return cursor
    conn = sqlite3.connect('database/deafening.db')
    # conn.row_factory = lambda cursor, row: row[0]
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if query:
        cur.execute(query[0])
    return cur, conn


def cell_info(conn, cluster_run):
    from types import SimpleNamespace
    project_path = project()
    df = pd.read_sql_query("SELECT * FROM cluster WHERE id = (?)", conn, params=(cluster_run,))
    df = df.applymap(str)
    this_dic = df.iloc[0].to_dict()
    cell = SimpleNamespace(**this_dic)

    if len(cell.id) == 1:
        cell.id = '00' + cell.id
    elif len(cell.id) == 2:
        cell.id = '0' + cell.id

    if len(cell.taskSession) == 1:
        cell.taskSession = 'D0' + str(cell.taskSession)
    elif len(cell.taskSession) == 2:
        cell.taskSession = 'D' + str(cell.taskSession)
    cell.site = cell.site[-2:]

    cell_name = [cell.id, cell.birdID, cell.taskName, cell.taskSession, cell.sessionDate, cell.channel, cell.unit]
    cell_name = '-'.join(cell_name)
    cell_path = os.path.join(project_path,cell.birdID, cell.taskName, cell.taskSession + '(' + cell.sessionDate + ')', cell.site, 'Songs')

    return cell, cell_name, cell_path


def song_info(conn, song_run):
    from types import SimpleNamespace
    project_path = project()
    song_run += 1
    # df = pd.read_sql_query("SELECT * FROM song WHERE id = (?)", conn, params=(song_run,))
    df = pd.read_sql_query("SELECT * FROM song", conn)
    df = df.applymap(str)
    this_dic = df.iloc[song_run].to_dict()
    song = SimpleNamespace(**this_dic)

    if len(song.id) == 1:
        song.id = '00' + song.id
    elif len(song.id) == 2:
        song.id = '0' + song.id

    if len(song.taskSession) == 1:
        song.taskSession = 'D0' + str(song.taskSession)
    elif len(song.taskSession) == 2:
        song.taskSession = 'D' + str(song.taskSession)

    song_name = [song.id, song.birdID, song.taskName, song.taskSession, song.sessionDate]
    song_name = '-'.join(song_name)
    song_path = os.path.join(project_path, song.birdID, song.taskName, song.taskSession + '(' + song.sessionDate + ')')

    return song, song_name, song_path




if __name__ == '__main__':
    cur = database()
