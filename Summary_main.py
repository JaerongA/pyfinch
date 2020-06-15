import os
import pandas as pd

projectROOT = r'H:\Box\Data\Deafening Project'  # home
# projectROOT = r'C:\Users\jahn02\Box\Data\Deafening Project'  # lab
# projectROOT =r'C:\Users\AJR\Box\Data\Deafening Project'  # home


summary_path = projectROOT + '\Analysis\Summary'
summary = "Cluster_summary(Deafening).xlsx"

os.chdir(summary_path)
summary_cluster = pd.read_excel(summary)

print('Loading the summary file')

nb_cluster = summary_cluster.shape[0]

for cluster_run in range(0, nb_cluster):


    ## Read information from cluster summary
    Key = str(summary_cluster['Key'][cluster_run])
    if len(Key) == 1: Key = '0' + str(Key) # make the length equal

    BirdID = summary_cluster['BirdID'][cluster_run]
    TaskName = summary_cluster['TaskName'][cluster_run]
    TaskSession = summary_cluster['TaskSession'][cluster_run]
    TaskSessionDeafening = summary_cluster['TaskSessionDeafening'][cluster_run]
    TaskSessionPostdeafening = summary_cluster['TaskSessionPostdeafening'][cluster_run]
    DPH = summary_cluster['DPH'][cluster_run]
    Block = summary_cluster['Block'][cluster_run]
    Block = summary_cluster['TaskName'][cluster_run]
    Block = summary_cluster['TaskName'][cluster_run]
    Block = summary_cluster['TaskName'][cluster_run]
    TaskName = summary_cluster['TaskName'][cluster_run]



    print(Key,BirdID)