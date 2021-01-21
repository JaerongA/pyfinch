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

    @property
    def open_folder(self):
        """Open the directory in win explorer"""
        import webbrowser
        webbrowser.open(self.path)

    def load_db(self):
        db_path = self.path / self.db_path / self.db
        return DBLoader(db_path)


class DBLoader:
    def __init__(self, db_path):

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()

    def execute(self, query):
        # get column names
        self.cur.execute(query)
        self.row = self.cur.fetchone()
        self.col_names = self.row.keys()
        self.cur.execute(query)

