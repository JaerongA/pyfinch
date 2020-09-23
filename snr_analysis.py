"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load
from load_intan_rhd_format.load_intan_rhd_format import read_data
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

query = "SELECT * FROM cluster WHERE id = '22'"
cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():
    cell_name, cell_path = load.cell_info(cell_info)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).mat'))[0]
    channel_info = scipy.io.loadmat(mat_file)
    # print(channel_info.keys())
    spk_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).txt'))[0]
    print(spk_file)


    def read_spk_file(spk_file: str):
        # Column header of the input .txt
        # ['Channel', 'Unit', 'Timestamp']
        # Column 3 to 35 stores waveforms

        spk_info = np.loadtxt(spk_file, delimiter='\t', skiprows=1)  # skip header
        spk_