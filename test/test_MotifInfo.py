import unittest
from pyfinch.analysis import MotifInfo
from pyfinch.database.load import ProjectLoader
from pyfinch.analysis import *
from pyfinch.analysis import *
from scipy.io import wavfile
from song.parameters import *
from pathlib import Path
from pyfinch.analysis import read_rhd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from pyfinch.utils.spect import *
from util.draw import *
import math


class TestMotifInfo(unittest.TestCase):

    query = "SELECT * FROM cluster WHERE id == 9"
    project = ProjectLoader()
    cur, conn, col_names = project.load_db(query)
    row = cur.fetchall()

    def test_len(self, row):

        mi = MotifInfo(row, update=True)
        len(mi)


if __name__ == "__main__":
    unittest.main()
