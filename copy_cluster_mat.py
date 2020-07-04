# Created by JR (06/29/2020)
# This program copy/pastes SpkInfo.mat from each cell root to the destinatoin folder (InformationAnalysis)

import os
from summary import load
from summary import save

config_file = 'summary/project.ini'
parser = load.config(config_file)
projectROOT = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del config_file, parser

# Make a folder to save files
saveROOT = 'InformationAnalysis'
save.make_save_dir(saveROOT)



saveROOT = projectROOT + '\\Analysis\\InformationAnalysis'
if not os.path.exists(saveROOT):
    os.mkdir(saveROOT)
df

def copy_cluster_mat(summary_cluster):
    import shutil

    for cluster_run in range(0, nb_cluster):
        cluster = load.cluster(summary_cluster, cluster_run)

        if int(cluster.AnalysisOK):
            # print(cluster)
            sessionID, cellID, cellROOT = load.cluster_info(cluster)
            print('Accessing... ' + cellROOT)
            os.chdir(cellROOT)

            mat_file = [file for file in os.listdir(cellROOT) if file.endswith('SpkInfo.mat')][0]
            print(mat_file)

            # Make a new folder for individual neurons
            saveROOT2 = saveROOT + '\\' + cellID
            if not os.path.exists(saveROOT2):
                os.mkdir(saveROOT2)

            shutil.copy(mat_file, saveROOT2)


if __name__ == '__main__':
    copy_cluster_mat(summary_cluster)
