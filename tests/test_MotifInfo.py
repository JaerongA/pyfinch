import math
import unittest
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.io import wavfile
from song.parameters import *
from util import save
from util.draw import *

from pyfinch.analysis import *
from pyfinch.analysis import MotifInfo, read_rhd
from pyfinch.database.load import ProjectLoader
from pyfinch.utils.spect import *


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
