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
from song.parameters import *
from utilities.functions import *
from spike.analysis import SpkInfo

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE id == 50"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():

    cell_name, cell_path = load.cell_info(row)
    print('Accessing... ' + cell_name)

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
    context_list = []
    bout_list = []
    onset_list = []

    # Loop through Intan .rhd files
    for rhd in rhd_files:
        # Load the .rhd file
        print('Loading... ' + rhd.stem  + '\n')
        # intan = read_rhd(rhd)
        # intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0

        # Load the .not.mat file
        notmat_file = rhd.with_suffix('.wav.not.mat')
        # print(notmat_file)

        onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file)

        # Find motifs
        motif_ind = find_str(row['motif'], syllables)

        # Get syllable, spike time stamps
        for ind in motif_ind:
            start_ind = ind
            stop_ind = ind + len(row['motif']) - 1

            motif_onset = onsets[start_ind]
            motif_offset = offsets[stop_ind]

            motif_spk = spk_ts[np.where((spk_ts >= motif_onset) & (spk_ts <= motif_offset))]

            nb_spk_vec.append(len(motif_spk))
            time_vec.append(motif_offset - motif_onset)
            context_list.append(context)


        # Demarcate song bouts

        def demarcate_bout(target, intervals):

            ind = np.where(intervals > bout_crit)[0]
            bout_labeling = target

            if isinstance(target, str):
                if len(ind):
                    for i, item in enumerate(ind):
                        if i is 0:
                            bout_labeling = target[:item + 1]
                        else:
                            bout_labeling += '*' + target[ind[i - 1] + 1:ind[i] + 1]
                    bout_labeling += '*' + target[ind[i] + 1:]

                bout_labeling += '*'  # end with an asterisk


            elif isinstance(target, np.ndarray):
                if len(ind):
                    for i, item in enumerate(ind):
                        if i is 0:
                            bout_labeling = target[:item + 1]
                        else:
                            bout_labeling = np.append(bout_labeling, '*')
                            bout_labeling = np.append(bout_labeling, target[ind[i - 1] + 1: ind[i] + 1])

                    bout_labeling = np.append(bout_labeling, target[ind[i] + 1:])
                    bout_labeling = np.append(bout_labeling, '*')  # end with an asterisk

            return bout_labeling



        bout_list.append(demarcate_bout(syllables, intervals))
        onset_list.append(demarcate_bout(onsets, intervals))



        # # Demarcate song bouts
        # ind = np.where(intervals > bout_crit)[0]
        # bout_labeling = syllables
        # if len(ind):
        #     for i, item in enumerate(ind):
        #         if i is 0:
        #             bout_labeling = syllables[:item + 1]
        #         else:
        #             bout_labeling += '*' + syllables[ind[i - 1] + 1:ind[i] + 1]
        #     bout_labeling += '*' + syllables[ind[i] + 1:]
        #
        # bout_labeling += '*'  # end with an asterisk
        # bout_list1.append(bout_labeling)
        #
        # # Demarcate song bouts
        # ind = np.where(intervals > bout_crit)[0]
        # bout_labeling = onsets
        # if len(ind):
        #     for i, item in enumerate(ind):
        #         if i is 0:
        #             bout_labeling = onsets[:item + 1]
        #         else:
        #             bout_labeling = np.append(bout_labeling, '*')
        #             bout_labeling = np.append(bout_labeling, onsets[ind[i - 1] + 1 : ind[i] + 1])
        #
        #     bout_labeling = np.append(bout_labeling, '*')
        #     bout_labeling = np.append(bout_labeling, onsets[ind[i] + 1:])
        #
        # bout_labeling = np.append(bout_labeling, '*')  # end with an asterisk
        # bout_list2.append(bout_labeling)








    # Calculate firing rates per condition
    fr = {'Undir':
              np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_list) if
                          context == 'U']).mean(),
          'Dir':
              np.asarray([nb_spk / time for nb_spk, time, context in zip(nb_spk_vec, time_vec, context_list) if
                          context == 'D']).mean()
          }

    break
