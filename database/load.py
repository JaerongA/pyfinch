"""
By Jaerong
Load project information and read from the project database
"""
from configparser import ConfigParser
import sqlite3
from pathlib import Path


class ProjectLoader:
    def __init__(self):
        config_file = 'config.ini'
        config = ConfigParser()
        config.read(config_file)
        self.path = Path(config.get('path', 'project'))
        self.db_path = Path(config.get('path', 'database'))
        self.db = Path(config.get('file', 'database'))
        self.parameter = config['parameter']
        # self.parameter = config.get('parameter')
        # print(config.sections())
        # print(config['path'])

    @property
    def open_folder(self):
        """Open the directory in win explorer"""
        import webbrowser
        webbrowser.open(self.path)

    def load_db(self, query):
        database_path = self.path / self.db_path / self.db
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # get column names
        cur.execute(query)
        row = cur.fetchone()
        col_names = row.keys()
        cur.execute(query)

        return cur, conn, col_names

    def load_parms(self):
        pass

# def cluster_info(row):
#     project_path = project()
#
#     if len(str(row['id'])) == 1:
#         cluster_id = '00' + str(row['id'])
#     elif len(str(row['id'])) == 2:
#         cluster_id = '0' + str(row['id'])
#
#     if len(str(row['taskSession'])) == 1:
#         cluster_taskSession = 'D0' + str(str(row['taskSession']))
#     elif len(str(row['taskSession'])) == 2:
#         cluster_taskSession = 'D' + str(str(row['taskSession']))
#
#     cluster_name = [cluster_id, row['birdID'], row['taskName'], cluster_taskSession, row['sessionDate'], row['site'],
#                  row['channel'], row['unit']]
#     cluster_name = '-'.join(map(str, cluster_name))
#
#     cluster_path = os.path.join(project_path, row['birdID'], row['taskName'],
#                              cluster_taskSession + '(' + str(row['sessionDate']) + ')', row['site'][-2:],
#                              'Songs')
#     cluster_path = Path(cluster_path)
#
#     return cluster_name, cluster_path
#
#
# def song_info(row):
#
#     project_path = project()
#
#     if len(str(row['id'])) == 1:
#         song_id = '00' + str(row['id'])
#     elif len(str(row['id'])) == 2:
#         song_id = '0' + str(row['id'])
#
#     if len(str(row['taskSession'])) == 1:
#         song_taskSession = 'D0' + str(str(row['taskSession']))
#     elif len(str(row['taskSession'])) == 2:
#         song_taskSession = 'D' + str(str(row['taskSession']))
#
#     song_name = [song_id, row['birdID'], row['taskName'], song_taskSession, row['sessionDate']]
#     song_name = '-'.join(map(str, song_name))
#     song_path = os.path.join(project_path, row['birdID'], row['taskName'],
#                              song_taskSession + '(' + str(row['sessionDate']) + ')')
#     song_path = Path(song_path)
#
#     return song_name, song_path
