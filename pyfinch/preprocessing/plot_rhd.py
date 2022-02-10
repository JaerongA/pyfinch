"""
load & plot the neural data (.rhd)
"""
import os
from util.load_intan_rhd_format import read_rhd
import matplotlib.pyplot as plt

# summary_df, nb_cluster = load.summary(load.config())  # load cluster summary file
#
# for cluster_run in range(0, nb_cluster):
#     cluster = load.cluster(summary_df, cluster_run)
#     session_id, cell_id, session_path, cell_path = load.cluster_info(cluster)
#
#     print('Accessing... ' + cell_path)
#     os.chdir(cell_path)
#     rhd_files = [file for file in os.listdir(cell_path) if file.endswith('.rhd')]
#
#     for rhd in rhd_files:
#         intan = read_rhd(rhd)  # load the .rhd file
#         # print(intan.keys())
#
#         plt.figure(figsize=(6, 4))
#         for ch in intan['amplifier_data']:
#             plt.plot(intan['t_amplifier'], ch, 'k')
#     break
#     plt.show()
#     # break


data_dir = r"\\rstore1.uit.tufts.edu\as_rsch_kao_lab01$\Data\jaerong\y61r65"
os.chdir(data_dir)
rhd_files = [file for file in os.listdir(data_dir) if file.endswith('.rhd')]

for rhd in rhd_files:
    intan = read_rhd(rhd)  # load the .rhd file
    # print(intan.keys())

    plt.figure(figsize=(6, 4))
    for ch in intan['amplifier_data']:
        plt.plot(intan['t_amplifier'], ch, 'k')
plt.show()