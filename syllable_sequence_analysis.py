import os
from summary import load
from song_analysis.parameters import *


summary_df, nb_cluster = load.summary(load.config())

for cluster_run in range(0, nb_cluster):

    cluster = load.cluster(summary_df, cluster_run)
    session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)
    print(session_path)
    print(bout_crit)
    break
