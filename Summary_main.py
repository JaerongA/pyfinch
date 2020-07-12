import os
import pandas as pd
from datetime import date
from summary import load_project

today = date.today()

project_path = r'H:\Box\Data\Deafening Project'  # home
# projectROOT = r'C:\Users\jahn02\Box\Data\Deafening Project'  # lab
# projectROOT =r'C:\Users\AJR\Box\Data\Deafening Project'  # home

summary_path = project_path + '\Analysis\Summary'
os.chdir(summary_path)

# Load summary
summary = "Cluster_summary(Deafening).xlsx"
summary_cluster = pd.read_excel(summary).applymap(str)  # read as a string

print('Loading the summary file')

# Save folder
saveROOT = project_path + '/Analysis/SyllableDuration/' + str(today)
if not (os.path.isdir(saveROOT)): os.mkdir(saveROOT)

nb_cluster = summary_cluster.shape[0]

for cluster_run in range(0, nb_cluster):

    # ## Read information from cluster summary
    # Key = summary_cluster['Key'][cluster_run]
    # # make the length equal
    # if len(Key) == 1: Key = '00' + Key
    # elif len(Key) == 2: Key = '0' + Key
    #
    # BirdID = summary_cluster['BirdID'][cluster_run]
    # TaskName = summary_cluster['TaskName'][cluster_run]
    # TaskSession = summary_cluster['TaskSession'][cluster_run]
    # if len(TaskSession) == 1: TaskSession = 'D0' + TaskSession
    # elif len(TaskSession) == 2: TaskSession = 'D' + TaskSession
    #
    # TaskSessionDeafening = summary_cluster['TaskSessionDeafening'][cluster_run]
    # TaskSessionPostdeafening = summary_cluster['TaskSessionPostdeafening'][cluster_run]
    # DPH = summary_cluster['DPH'][cluster_run]
    # Block = summary_cluster['Block_10days'][cluster_run]
    # SessionDate = summary_cluster['SessionDate'][cluster_run]
    # Site = summary_cluster['Site'][cluster_run][-2:]
    # Channel = summary_cluster['Channel'][cluster_run]
    # Cluster = summary_cluster['Cluster'][cluster_run]
    # ClusterQuality = summary_cluster['ClusterQuality'][cluster_run]
    # SongNote = summary_cluster['SongNote'][cluster_run]
    # Motif = summary_cluster['Motif'][cluster_run]
    # IntroNotes = summary_cluster['IntroNotes'][cluster_run]
    # Calls = summary_cluster['Calls'][cluster_run]
    #
    # SessionID = Key + '-' + BirdID + '-' + TaskName + '-' + TaskSession + '-' + SessionDate + '-Site' + Site
    # CellID = Key + '-' + BirdID + '-' + TaskName + '-' + TaskSession + '-' + SessionDate + '-Site' + Site + '-Ch' + Channel +'-Cluster' + Cluster
    # dataROOT = projectROOT + '/' + BirdID + '/' + TaskName + '/' + TaskSession + '(' + SessionDate + ')/' + Site + '/Songs'
    # print('Accessing' + dataROOT)
    # SessionID = Key + '-' + BirdID + '-' + TaskName + '-' + TaskSession + '-' + SessionDate + '-Site' + Site
    # os.chdir(dataROOT)

    this_dic = summary_cluster.iloc[cluster_run].to_dict()

    # # for key, value in this_dic.items():
    # #     key = value
    #
    # # for key, value in this_dic.items():
    # #     exec('{} = {}'.format(key, value))

    # >> > from types import SimpleNamespace
    # >> > d = {'a': 1, 'b': 2}
    # >> > n = SimpleNamespace(**d)
    # >> > n.a
    # 1


    locals().update(this_dic)
    # print(os.listdir(dataROOT))
    break
