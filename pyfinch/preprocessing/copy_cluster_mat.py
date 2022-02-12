"""
This program copy/pastes SpkInfo.mat from each cell root to the destination folder (InformationAnalysis)
"""

import os
from summary import load

project_path = load.project(load.config())  # load cluster summary file
summary_df, nb_cluster = load.summary(load.config())  # load cluster summary file

# Make a folder to save files
# # save_path = 'InformationAnalysis'
# # save.make_save_dir(save_path)


save_path = os.path.join(project_path,
                         r'Analysis\InformationAnalysis')  # the data folder where SAP feature values are stored
if not os.path.exists(save_path):
    os.mkdir(save_path)


def copy_cluster_mat(summary_df):
    import shutil

    for cluster_run in range(0, nb_cluster):
        cluster = load.cluster(summary_df, cluster_run)

        if int(cluster.AnalysisOK):
            # print(cluster)
            session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)
            # print('Accessing... ' + cell_root)
            os.chdir(cell_path)

            mat_file = [file for file in os.listdir(cell_path) if file.endswith('SpkInfo.mat')][0]
            # print(mat_file)

            # Make a new folder for individual neurons
            new_save_path = os.path.join(save_path, cell_id)
            print(new_save_path)
            if not os.path.exists(new_save_path):
                os.mkdir(new_save_path)
            shutil.copy(mat_file, new_save_path)


if __name__ == '__main__':
    copy_cluster_mat(summary_df)
