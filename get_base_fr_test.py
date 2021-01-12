from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
from util import save

query = "SELECT * FROM cluster WHERE id == 65"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)

for row in cur.fetchall():

    #
    # # ci = ClusterInfo(row)
    #
    # print(ci)

    bi = BaselineInfo(row)
    break