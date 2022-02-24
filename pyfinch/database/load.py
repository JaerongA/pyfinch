"""
Load project information and read from the project database
"""

import sqlite3
from configparser import ConfigParser


def create_db(sql_file_name: str):
    """
    Create a new sql table from .slq file in /database

    Parameters
    ----------
    sql_file_name : str
        name of the sql file to create
    """

    # Load database
    db = ProjectLoader().load_db()
    # Make database
    with open(sql_file_name, 'r') as sql_file:
        db.conn.executescript(sql_file.read())


class ProjectLoader:

    def __init__(self):
        from pathlib import Path

        config_file = Path(__file__).resolve().parent.parent / 'config.ini'
        config = ConfigParser()
        config.read(config_file)
        self.path = Path(config.get('path', 'project'))
        self.db_path = Path(config.get('path', 'database'))
        self.db = Path(config.get('file', 'database'))

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

        self.cur.execute(query)

    def create_col(self, table: str, col_name: str, type: str):
        """
        Create a new column in table

        Parameters
        ----------
        table : str
            table name to add the column
        col_name : str
            column name
        type : str
            data type for the column (e.g., TEXT, INT)
        """
        if col_name not in self.col_names(table):
            self.cur.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, col_name, type))

    def col_names(self, table) -> list:
        """Get column names"""
        self.cur.execute(f"PRAGMA table_info({table})")
        columns = self.cur.fetchall()
        return [x[1] for x in columns]

    def update(self, table, col_name, condition_col=None, condition_value=None, value=None):
        """Update values to table"""
        if condition_col is not None:
            self.cur.execute("UPDATE {} SET {} = ? WHERE {} = ?".format(table, col_name, condition_col),
                             (value, condition_value))
        else:
            self.cur.execute("UPDATE {} SET {} = ? WHERE {} = ?".format(table, col_name, condition_col),
                             (value, condition_value))
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
        from pathlib import Path

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
        """
        Convert to pandas dataframe according to the query statement

        Parameters
        ----------
        query : str
            SQL query statement

        Returns
        -------
        df : dataframe
        """
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

        from pathlib import Path

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
        cluster_path = project_path / self.birdID / self.taskName / \
                       cluster_taskSession / self.site[-2:] / 'Songs'
        cluster_path = Path(cluster_path)
        return cluster_name, cluster_path

    def load_song_db(self):

        from pathlib import Path

        song_id = ''
        if len(str(self.id)) == 1:
            song_id = '00' + str(self.id)
        elif len(str(self.id)) == 2:
            song_id = '0' + str(self.id)
        else:
            song_id = str(self.id)

        task_session = ''
        if len(str(self.taskSession)) == 1:
            task_session = 'D0' + str(self.taskSession)
        elif len(str(self.taskSession)) == 2:
            task_session = 'D' + str(self.taskSession)
        task_session += '(' + str(self.sessionDate) + ')'

        song_name = [song_id, self.birdID, self.taskName, task_session]
        song_name = '-'.join(map(str, song_name))

        # Get cluster path
        project_path = ProjectLoader().path
        song_path = project_path / self.birdID / self.taskName / task_session
        song_path = Path(song_path)
        return song_name, song_path
