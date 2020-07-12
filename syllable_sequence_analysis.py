import os
from summary.read_config import parser
from summary import load
from summary import save
from song_analysis.parameters import *

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del parser


for cluster_run in range(0, nb_cluster):

    cluster = load.cluster(summary_cluster, cluster_run)
    session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)
    print(session_path)
    print(bout_crit)
    break
