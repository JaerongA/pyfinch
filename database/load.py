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
        self.cur.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, col_name, type))

    def update(self, table, col_name, type, id, value=None):

        if col_name not in self.col_names:
            self.create_col(table, col_name, type)
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
            """Open the directory in win explorer"""
            import webbrowser
            webbrowser.open(self.dir)