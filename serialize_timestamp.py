"""
By Jaerong
Serialize timestamps across multiple files recorded in the same session
"""

from database import load
import numpy as np
from pathlib import Path
from spike.load import read_spk_txt, read_rhd
from spike.parameters import *
from spike.analysis import *
from song.functions import *
from song.parameters import *
from utilities.functions import *

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE id == 50 "
#query = "SELECT * FROM cluster WHERE analysisOK IS TRUE"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():

    cell_name, cell_path = load.cluster_info(row)
    print('Accessing... ' + cell_name + '\n')

    # # List .rhd files
    # rhd_files = list(cell_path.glob('*.rhd'))
    #
    # # Initialize variables
    # t_amplifier_serialized = np.array([], dtype=np.float64)
    #
    # # Store values in these lists
    # file_list = []
    # file_start_list = []
    # file_end_list = []
    # onset_list = []
    # offset_list = []
    # syllable_list = []
    # context_list = []
    #
    # # Loop through Intan .rhd files
    # for file in rhd_files:
    #
    #     # Load the .rhd file
    #     print('Loading... ' + file.stem)
    #     intan = read_rhd(file)  # note that the timestamp is in second
    #
    #     # Load the .not.mat file
    #     notmat_file = file.with_suffix('.wav.not.mat')
    #     onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file, unit='ms')
    #
    #     # Serialize time stamps
    #     intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0
    #     start_ind = t_amplifier_serialized.size  # start of the file
    #
    #     if t_amplifier_serialized.size == 0:
    #         t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])
    #     else:
    #         intan['t_amplifier'] += (t_amplifier_serialized[-1] + (1/sample_rate['intan']))
    #         t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])
    #
    #     # File information (name, start and end timestamp of each file)
    #     file_list.append(file.stem)
    #     file_start_list.append(t_amplifier_serialized[start_ind] * 1E3) # in ms
    #     file_end_list.append(t_amplifier_serialized[-1] * 1E3)
    #
    #     onsets += intan['t_amplifier'][0] *1E3  # convert to ms
    #     offsets += intan['t_amplifier'][0] *1E3
    #
    #     #Demarcate song bouts
    #     onset_list.append(demarcate_bout(onsets, intervals))
    #     offset_list.append(demarcate_bout(offsets, intervals))
    #     syllable_list.append(demarcate_bout(syllables, intervals))
    #     context_list.append(context)
    #
    # # Organize event-related info into a single dictionary object
    # event_info = {
    #     'file' : file_list,
    #     'file_start' : file_start_list,
    #     'file_end': file_end_list,
    #     'onsets' :  onset_list,
    #     'offsets' : offset_list,
    #     'syllables': syllable_list,
    #     'context' : context_list
    # }

    event_info = get_event_info(cell_path)
    break


    ## Get baseline firing rates
    # Get the cluster .txt file
    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    # Read from the cluster .txt file
    unit_nb = int(row['unit'][-2:])
    spk_ts, _, _ = read_spk_txt(spk_txt_file, unit_nb, unit='ms')


    baseline_spk_vec = []
    nb_spk_vec = []
    time_vec = []

    for file_ind, (onset, offset, syllable, file_start) in \
            enumerate(zip(event_info['onsets'], event_info['offsets'], event_info['syllables'], event_info['file_start'])):

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

            if baseline_offset < file_start:  # skip if there's not enough baseline period at the start of a file
                continue
            elif bout_ind > 0 and baseline_onset < float(
                    offset[bout_ind - 1]):  # skip if the baseline starts before the offset of the previous syllable
                continue
            elif baseline_onset < file_start:
                baseline_onset = file_start

            baseline_spk = spk_ts[np.where((spk_ts >= baseline_onset) & (spk_ts <= baseline_offset))]

            baseline_spk_vec.append(baseline_spk)
            nb_spk_vec.append(len(baseline_spk))
            time_vec.append((baseline_offset - baseline_onset) / 1E3)  # convert to seconds for calculating in Hz


    # Calculate baseline firing rates
    fr = {'Baseline':sum(nb_spk_vec)/sum(time_vec)}

    # Get motif firing rates
    motif_spk_vec = []
    nb_spk_vec = []
    time_vec = []
    context_vec = []

    for file_ind, (onset, offset, syllable, context) in \
            enumerate(zip(event_info['onsets'], event_info['offsets'], event_info['syllables'], event_info['context'])):

        # Find motifs
        motif_ind = find_str(row['motif'], syllable)

        # Get syllable, spike time stamps
        for ind in motif_ind:
            start_ind = ind
            stop_ind = ind + len(row['motif']) - 1

            motif_onset = float(onset[start_ind])
            motif_offset = float(offset[stop_ind])

            motif_spk = spk_ts[np.where((spk_ts >= motif_onset) & (spk_ts <= motif_offset))]

            motif_spk_vec.append(motif_spk)
            nb_spk_vec.append(len(motif_spk))
            time_vec.append((motif_offset - motif_onset)/1E3)  # convert to seconds for calculating in Hz
            # Store context info
            context_vec.append(context)


    # Calculate motif firing rates per condition
    # if not bool(fr):  # if fr dictionary does not exist
    #     if ''.join(context_vec).find('U') >= 0:  # if undir exists
    #         fr = {'MotifUndir' :
    #                   sum([nb_spk for nb_spk, context in zip(nb_spk_vec, context_vec) if context == 'U'])\
    #                   / sum([time for time, context in zip(time_vec, context_vec) if context == 'U'])
    #               }
    #     elif
    #           'MotifDir' :
    #               sum([nb_spk for nb_spk, context in zip(nb_spk_vec, context_vec) if context == 'D'])\
    #               / sum([time for time, context in zip(time_vec, context_vec) if context == 'D'])
    #           }
    # else:

    if ''.join(context_vec).find('U') >= 0:  # if undir exists
        fr['MotifUndir'] = sum([nb_spk for nb_spk, context in zip(nb_spk_vec, context_vec) if context == 'U'])\
                  / sum([time for time, context in zip(time_vec, context_vec) if context == 'U'])

    if ''.join(context_vec).find('D') >= 0:  # if undir exists
        fr['MotifDir'] = sum([nb_spk for nb_spk, context in zip(nb_spk_vec, context_vec) if context == 'D'])\
                  / sum([time for time, context in zip(time_vec, context_vec) if context == 'D'])


    # Update the database
    if bool('Baseline' in fr):
        cur.execute("UPDATE cluster SET baselineFR= ? WHERE id = ?", (format(fr['Baseline'], '.3f'), row['id']))
    if bool('MotifUndir' in fr):
        cur.execute("UPDATE cluster SET motifFRUndir = ? WHERE id = ?", (format(fr['MotifUndir'], '.3f'), row['id']))
    if bool('MotifDir' in fr):
        cur.execute("UPDATE cluster SET motifFRDir = ? WHERE id = ?", (format(fr['MotifDir'], '.3f'), row['id']))
    conn.commit()
    # break


