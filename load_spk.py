import os
from summary.read_config import parser
from summary import load
from summary import save
from load_intan_rhd_format.load_intan_rhd_format import read_data

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file

for cluster_run in range(0, nb_cluster):
    cluster = load.cluster(summary_cluster, cluster_run)
    sessionID, cellID, cellROOT = load.cluster_info(cluster)

    print('Accessing... ' + cellROOT)
    os.chdir(cellROOT)
    rhd_files = [file for file in os.listdir(cellROOT) if file.endswith('.rhd')]

    for rhd in rhd_files:
        a =  read_data(rhd)  # load the .rhd file
    break





