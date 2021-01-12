"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""

from database import load
import numpy as np
from pathlib import Path
from analysis.load import read_spk_txt, read_rhd
from analysis.spike import SpkInfo
from analysis.parameters import *
from song.analysis import *
from song.parameters import *
from util.functions import *

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE id == 50"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():

    cell_name, cell_path = load.cluster_info(row)
    print('Accessing... ' + cell_name + '\n')

    # Get the cluster .txt file
    unit_nb = int(row['unit'][-2:])
    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    # Read from the cluster .txt file
    spk_ts, _, _ = read_spk_txt(spk_txt_file, unit_nb, unit='ms')

    # List .rhd files
    rhd_files = list(cell_path.glob('*.rhd'))

    # Initialize variables
    t_amplifier_serialized = np.array([], dtype=np.float64)

    nb_spk_vec = []
    time_vec = []
    context_vec = []

    onset_list = []
    offset_list = []
    syllable_list = []
    context_list = []

    # Loop through Intan .rhd files
    for rhd in rhd_files:

        # Load the .rhd file
        print('Loading... ' + rhd.stem)
        # intan = read_rhd(rhd)
        # intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0

        # Load the .not.mat file
        notmat_file = rhd.with_suffix('.wav.not.mat')
        # print(notmat_file)
        onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file)

        # Find motifs
        motif_ind = find_str(row['motif'], syllables)

        # Get syllable, analysis time stamps
        for ind in motif_ind:
            start_ind = ind
            stop_ind = ind + len(row['motif']) - 1

            motif_onset = onsets[start_ind]
            motif_offset = offsets[stop_ind]

            motif_spk = spk_ts[np.where((spk_ts >= motif_onset) & (spk_ts <= motif_offset))]

            nb_spk_vec.append(len(motif_spk))
            time_vec.append((motif_offset - motif_onset)/1E3)  # convert to seconds for calculating in Hz
            # Store context info
            context_vec.append(context)

        # Demarcate song bouts
        # onset_list.append(demarcate_bout(onsets, intervals))
        # offset_list.append(demarcate_bout(offsets, intervals))
        # syllable_list.append(demarcate_bout(syllables, intervals))
        # context_list.append(context)




    # Get baseline firing rates
    baseline_spk_vec = []
    nb_spk_vec = []
    time_vec = []

    for file_ind, (onset, offset, syllable) in enumerate(zip(onset_list, offset_list, syllable_list)):

        baseline_spk = []
        bout_ind_list = find_str('*', syllable_list[file_ind])
        bout_ind_list.insert(0, -1)  # start from the first index

        print('onset = {}'.format(onset))

        for bout_ind in bout_ind_list:
            print(bout_ind)
            if bout_ind == len(syllable) - 1:  # skip if * indicates the end syllable
                continue
            baseline_onset = float(onset[bout_ind + 1]) - baseline['time_buffer'] - baseline['time_win']
            baseline_offset = float(onset[bout_ind + 1]) - baseline['time_buffer']

            if baseline_offset < 0:  # skip if there's not enough baseline period at the start of a file
                continue
            elif bout_ind > 0 and baseline_onset < float(
                    offset[bout_ind - 1]):  # skip if the baseline starts before the offset of the previous syllable
                continue
            elif baseline_onset < 0:
                baseline_onset = 0

            baseline_spk = spk_ts[np.where((spk_ts >= baseline_onset) & (spk_ts <= baseline_offset))]

            baseline_spk_vec.append(baseline_spk)
            nb_spk_vec.append(len(baseline_spk))
            time_vec.append((baseline_offset - baseline_onset) / 1E3)  # convert to seconds for calculating in Hz

    break




    # # Calculate motif firing rates per condition
    # fr = {'Undir':
    #           np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_vec) if
    #                       context == 'U']).mean(),
    #       'Dir':
    #           np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_vec) if
    #                       context == 'D']).mean()
    #       }
    #
    #
    #
    #
    # # Number of song bouts
    # nb_bouts = get_nb_bouts(row['songNote'], ''.join(syllable_list))
    #




    break
