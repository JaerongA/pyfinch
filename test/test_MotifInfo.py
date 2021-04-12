import unittest
from analysis.spike import MotifInfo
from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
from scipy.io import wavfile
from song.parameters import *
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from util.spect import *
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


if __name__ == '__main__':
    unittest.main()
