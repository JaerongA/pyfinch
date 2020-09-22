"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load
from load_intan_rhd_format.load_intan_rhd_format import read_data
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


query = "SELECT * FROM cluster WHERE id = '22'"
cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():
    cell_name, cell_path = load.cell_info(cell_info)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + cell_info['channel'] + '(merged).mat'))
    channel_info = scipy.io.loadmat(mat_file[0])
    print(channel_info)
    
