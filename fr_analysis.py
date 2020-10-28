"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""

from load_intan_rhd_format.load_intan_rhd_format import read_rhd
from database import load
import numpy as np
from pathlib import Path
from spike.load import read_spk_txt
from song.functions import read_not_mat

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE id == 6"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():

    cell_name, cell_path = load.cell_info(row)
    print('Loading... ' + cell_name)

    # Get the cluster .txt file
    unit_nb = int(row['unit'][-2:])
    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    # Read from the cluster .txt file
    spk_ts, _, _ = read_spk_txt(spk_txt_file, unit_nb)

    # List .rhd files
    rhd_files = list(cell_path.glob('*.rhd'))
    for rhd in rhd_files:

        # Load the .rhd file
        print('Loading... ' + rhd.name)
        intan = read_rhd(rhd)

        # Load the .not.mat file
        notmat_file = rhd.with_suffix('.wav.not.mat')
        print(notmat_file)
        onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file)
        break
