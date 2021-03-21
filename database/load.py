"""
By Jaerong
Load project information and read from the project database
"""
import sqlite3
from configparser import ConfigParser
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
        return Database(db_path)


class Database:
    def __init__(self, db_path):

        self.path = db_path
        self.dir = self.path.parent
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()

    def execute(self, query):
        # Get column names
        self.cur.execute(query)
        self.row = self.cur.fetchone()
        self.col_names = self.row.keys()
        self.cur.execute(query)

    def create_col(self, table, col_name, type):
        """
        Create a new column
        Args:
            table: str
                db table name
            col_name: str
                name of the column you like to create
            type: str
                data type for the column (e.g, TEXT, INT)

        """
        if col_name not in self.col_names:
            self.cur.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, col_name, type))

    def update(self, table, col_name, id, value=None):

        if value:
            self.cur.execute("UPDATE {} SET {} = ? WHERE id = ?".format(table, col_name), (value, id))
            self.conn.commit()

    def to_csv(self, table, add_date=True, open_folder=True):
        """
        Convert database to csv
        Parameters
        ----------
        table : str
            Name of the table
        open_folder : bool
            Open the output directory
        """
        from datetime import datetime
        import pandas as pd

        csv_name = Path(f"{table}.csv")
        if add_date:  # append time&time info to csv
            csv_name = csv_name.stem + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + csv_name.suffix

        df = pd.read_sql("SELECT * from {}".format(table), self.conn)
        df.to_csv(self.dir / csv_name, index=False, header=True)

        if open_folder:
            # Open the directory in win explorer
            import webbrowser
            webbrowser.open(self.dir)

    def to_dataframe(self, query):
        import pandas as pd
        df = pd.read_sql_query(query, self.conn)
        return df

class DBInfo:
    def __init__(self, db):
        # Set all database fields as attributes

        self.channel = None
        for key in db.keys():
            # dic[col] = database[col]
            setattr(self, key, db[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def load_cluster_db(self):
        """
        Return the list of files in the current directory
            Input: SQL object (database row)
            Output: name of the cluster
        """
        cluster_id = ''
        if len(str(self.id)) == 1:
            cluster_id = '00' + str(self.id)
        elif len(str(self.id)) == 2:
            cluster_id = '0' + str(self.id)
        else:
            cluster_id = str(self.id)

        cluster_taskSession = ''
        if len(str(self.taskSession)) == 1:
            cluster_taskSession = 'D0' + str(self.taskSession)
        elif len(str(self.taskSession)) == 2:
            cluster_taskSession = 'D' + str(self.taskSession)
        cluster_taskSession += '(' + str(self.sessionDate) + ')'

        if self.channel:  # if neural signal exists
            cluster_name = [cluster_id, self.birdID, self.taskName, cluster_taskSession,
                            self.site, self.channel, self.unit]
        else:
            cluster_name = [cluster_id, self.birdID, self.taskName, cluster_taskSession, self.site]

        cluster_name = '-'.join(map(str, cluster_name))

        # Get cluster path
        project_path = ProjectLoader().path
        cluster_path = project_path / self.birdID / self.taskName /\
                       cluster_taskSession / self.site[-2:] / 'Songs'
        cluster_path = Path(cluster_path)
        return cluster_name, cluster_path

    def load_song_db(self):
        pass