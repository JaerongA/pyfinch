# By Jaerong (06/29/2020)
# This program copy/pastes SpkInfo.mat from each cell root to the destinatoin folder (InformationAnalysis)

import os
from summary.read_config import parser
from summary import load
from summary import save

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del parser

# Make a folder to save files
save_path = 'InformationAnalysis'
save.make_save_dir(save_path)


save_path = os.path.join(project_path, r'Analysis\InformationAnalysis')  # the data folder where SAP feature values are stored
if not os.path.exists(save_path):
    os.mkdir(save_path)


def copy_cluster_mat(summary_cluster):

    import shutil

    for cluster_run in range(0, nb_cluster):
        cluster = load.cluster(summary_cluster, cluster_run)

        if int(cluster.AnalysisOK):
            # print(cluster)
            sessionID, cellID, cellROOT = load.cluster_info(cluster)
            # print('Accessing... ' + cellROOT)
            os.chdir(cellROOT)

            mat_file = [file for file in os.listdir(cellROOT) if file.endswith('SpkInfo.mat')][0]
            # print(mat_file)

            # Make a new folder for individual neurons
            save_path_new = os.path.join(save_path, cellID)
            print(save_path_new)
            if not os.path.exists(save_path_new):
                os.mkdir(save_path_new)

            shutil.copy(mat_file, save_path_new)


if __name__ == '__main__':
    copy_cluster_mat(summary_cluster)
