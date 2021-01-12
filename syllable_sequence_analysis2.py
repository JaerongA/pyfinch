"""
By Jaerong
performs syllable sequence analysis and calculates transition entropy
"""

import os
from summary import load
from analysis.parameters import *
import scipy.io
import numpy as np


def is_song_bout(song_notes, bout):
    # returns the number of song notes within a bout
    nb_song_note_in_bout = len([note for note in song_notes if note in bout])
    return nb_song_note_in_bout


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


summary_df, nb_cluster = load.summary(load.config())
# Aggregate the dataframe (some days had recordings from multiple sites)
# summary_df = summary_df.groupby(['BirdID', 'TaskName', 'TaskSession'], sort=False).agg(','.join).reset_index()


# conditional select (optional)
# summary_df = summary_df.query('Key == "3" or Key == "4"')
# summary_df = summary_df.query('Key == "4"')

pre_path = ''
# context_list = list()
# bout_list = list()

for cluster_run in range(0, summary_df.shape[0]):

    cluster = load.cluster(summary_df, cluster_run)

    session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)

    if pre_path != session_path:
        context_list = list()
        bout_list = list()

    print(f'\nAccessing.........  {cell_path}\n')
    os.chdir(cell_path)

    mat_file = [file for file in os.listdir(cell_path) if file.endswith('.not.mat')]

    for file in mat_file:

        # load .not.mat
        print(file)
        syllables = scipy.io.loadmat(file)['syllables'][0]  # Load the syllable info
        onsets = scipy.io.loadmat(file)['onsets'].transpose()[0]  # syllable onset timestamp
        offsets = scipy.io.loadmat(file)['offsets'].transpose()[0]  # syllable offset timestamp
        intervals = onsets[1:] - offsets[:-1]  # interval among syllables
        context_list.append(file.split('.')[0].split('_')[-1][0].upper())  # extract 'U' or 'D' from the file name
        # print(intervals)

        # demarcate the song bout with an asterisk (stop)
        ind = np.where(intervals > bout_crit)[0]
        bout_labeling = syllables
        if len(ind):
            for i, item in enumerate(ind):
                if i is 0:
                    bout_labeling = syllables[:item + 1]
                else:
                    bout_labeling += '*' + syllables[ind[i - 1] + 1:ind[i] + 1]
            bout_labeling += '*' + syllables[ind[i] + 1:]

        bout_labeling += '*'
        # print(bout_labeling)
        bout_list.append(bout_labeling)

        # count the number of bouts (only include those having a song motif)
        nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if cluster.Motif in bout])

    print(bout_list)

    # Store song bouts and its context in a dict
    bout = {'Undir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'U']),
            'Dir': ''.join([bout for bout, context in zip(bout_list, context_list) if context == 'D'])}

    bout = {'Undir': {'notes': bout['Undir'],
                      'nb_bout': len([bout for bout in bout['Undir'].split('*')[:-1] if is_song_bout(cluster.SongNote, bout)])},
            'Dir': {'notes': bout['Dir'],
                    'nb_bout': len([bout for bout in bout['Dir'].split('*')[:-1] if is_song_bout(cluster.SongNote, bout)])}
            }

    # {'Undir': {'bout': 'kiiiiabcdjiabcdjiabcd*iiiabcdk*iiii*',
    #   'nb_bout': 2},
    #  'Dir': {'bout': 'kiiiiabcdjiabcdjiabcd*iiiabcdk*iiii*iiiabcdjiabcdk*kiiiiiabcdjia*',
    #   'nb_bout': 4}}

    pre_path = session_path

    # Save the results in .json format in the session path
    os.chdir(session_path)
    save_bout('config.json', bout)
