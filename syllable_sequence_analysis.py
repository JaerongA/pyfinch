"""
By Jaerong
performs syllable sequence analysis and calculates transition entropy
"""

import os
from summary import load
from song_analysis.parameters import *
import scipy.io
import numpy as np

summary_df, nb_cluster = load.summary(load.config())

# conditional select (optional)
summary_df = summary_df.query('Key == "1"')
pre_path = ''
context_list = list()
bout_list = list()

for cluster_run in range(0, summary_df.shape[0]):

    cluster = load.cluster(summary_df, cluster_run)

    session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)

    if pre_path is not session_path:
        context_list = list()
        bout_list = list()

    print('Accessing...... ' + cell_path)
    os.chdir(cell_path)

    mat_file = [file for file in os.listdir(cell_path) if file.endswith('.not.mat')]

    for file in mat_file:

        # load .not.mat
        print(file)
        syllables = scipy.io.loadmat(file)['syllables'][0]  # Load the syllable info
        onsets = scipy.io.loadmat(file)['onsets'].transpose()[0]  # syllable onset timestamp
        offsets = scipy.io.loadmat(file)['offsets'].transpose()[0]  # syllable offset timestamp
        intervals = onsets[1:] - offsets[:-1]  # interval among syllables
        context_list.append(file.split('.')[0][-3:][0])  # 'U' or 'D'
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
        print(bout_labeling)
        bout_list.append(bout_labeling)

        # count the number of bouts (only include those having a song motif)
        nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if cluster.Motif in bout])

    pre_path = session_path
