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
from song.functions import *
from utilities.functions import *

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE id == 50"
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

    # Initialize variables
    t_amplifier_serialized = np.array([], dtype=np.float64)

    nb_spk_vec = list()
    time_vec = list()
    context_list = list()

    # Loop through Intan .rhd files
    for rhd in rhd_files:
        # Load the .rhd file
        print('Loading... ' + rhd.name)
        intan = read_rhd(rhd)

        intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0

        # Load the .not.mat file
        notmat_file = rhd.with_suffix('.wav.not.mat')
        print(notmat_file)

        onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file, unit='second')

        # Find motifs
        motif_ind = find_str(row['motif'], syllables)

        # Get the time stamps
        for ind in motif_ind:

            start_ind = ind
            stop_ind = ind + len(row['motif']) - 1

            motif_onset = onsets[start_ind]
            motif_offset = offsets[stop_ind]

            motif_spk = spk_ts[np.where((spk_ts >= motif_onset) & (spk_ts <= motif_offset))]

            nb_spk_vec.append(len(motif_spk))
            time_vec.append(motif_offset - motif_onset)
            context_list.append(context)

    # Calculate firing rates per condition
    fr = {'Undir' :
              np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_list) if
                          context == 'U']).mean(),
          'Dir' :
              np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_list) if
                          context == 'D']).mean()
    }