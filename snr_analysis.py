"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load
from load_intan_rhd_format.load_intan_rhd_format import read_data
import pandas as pd
import numpy as np
import scipy.io
from spike.load import read_spk_txt
import matplotlib.pyplot as plt

query = "SELECT * FROM cluster WHERE id = '22'"
cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():
    cell_name, cell_path = load.cell_info(cell_info)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).mat'))[0]
    channel_info = scipy.io.loadmat(mat_file)
    spk_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).txt'))[0]
    unit_nb = int(cell_info['unit'][-2:])

    # Read from the cluster .txt file
    spk_ts, spk_waveform, nb_spk = read_spk_txt(spk_file, unit_nb)

    for wave in spk_waveform:
        print(wave.shape)
        break