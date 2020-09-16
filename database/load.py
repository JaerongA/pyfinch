"""
By Jaerong
Load project information and read from the project database
"""
import sqlite3
import pandas as pd
import os
from pathlib import Path


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


def database(query):
    # Apply query to the database
    # Return cursor, connection, col_name
    conn = sqlite3.connect('database/deafening.db')
    # conn.row_factory = lambda cursor, row: row[0]
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # get column names
    cur.execute(query)
    row = cur.fetchone()
    col_names = row.keys()

    cur.execute(query)

    return cur, conn, col_names


def cell_info(row):
    project_path = project()

    if len(str(row['id'])) == 1:
        cell_id = '00' + str(row['id'])
    elif len(str(row['id'])) == 2:
        cell_id = '0' + str(row['id'])

    if len(str(row['taskSession'])) == 1:
        cell_taskSession = 'D0' + str(str(row['taskSession']))
    elif len(str(row['taskSession'])) == 2:
        cell_taskSession = 'D' + str(str(row['taskSession']))

    cell_name = [cell_id, row['birdID'], row['taskName'], cell_taskSession, row['sessionDate'],
                 row['channel'], row['unit']]
    cell_name = '-'.join(map(str, cell_name))

    cell_path = os.path.join(project_path, row['birdID'], row['taskName'],
                             cell_taskSession + '(' + str(row['sessionDate']) + ')', row['site'][-2:],
                             'Songs')
    cell_path = Path(cell_path)

    return cell_name, cell_path


def song_info(row):

    project_path = project()

    if len(str(row['id'])) == 1:
        song_id = '00' + str(row['id'])
    elif len(str(row['id'])) == 2:
        song_id = '0' + str(row['id'])

    if len(str(row['taskSession'])) == 1:
        song_taskSession = 'D0' + str(str(row['taskSession']))
    elif len(str(row['taskSession'])) == 2:
        song_taskSession = 'D' + str(str(row['taskSession']))

    song_name = [song_id, row['birdID'], row['taskName'], song_taskSession, row['sessionDate']]
    song_name = '-'.join(map(str, song_name))
    song_path = os.path.join(project_path, row['birdID'], row['taskName'],
                             song_taskSession + '(' + str(row['sessionDate']) + ')')
    song_path = Path(song_path)

    return song_name, song_path


if __name__ == '__main__':
    cur = database()
