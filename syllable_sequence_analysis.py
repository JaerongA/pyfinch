import os
from summary import load
from song_analysis.parameters import *
import scipy.io

summary_df, nb_cluster = load.summary(load.config())

for cluster_run in range(0, nb_cluster):

    cluster = load.cluster(summary_df, cluster_run)
    session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)
    print('Accessing... ' + cell_path)
    os.chdir(cell_path)

    mat_file = [file for file in os.listdir(cell_path) if file.endswith('.not.mat')]

    for file in mat_file:
        syllables = scipy.io.loadmat(file)['syllables']  # Load the syllable info






        print(syllables)


    # mat_file = scipy.io.loadmat(mat_file)['mat_file']
    # print(mat_file)


    break
