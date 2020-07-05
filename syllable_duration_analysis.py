# By Jaerong (06/04/2020)
# Syllable duration for all syllables regardless of it type
# Calculation based on EventInfo.m

import os
import pandas as pd
import sys
from summary.read_config import parser
from summary import load
from summary import save
from datetime import date
import scipy.io
import subprocess

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del parser

path_name = 'SyllableDuration'
today = date.today().strftime("%Y-%m-%d")  # 2020-07-04
save_path = os.path.join(project_path, r'Analysis', path_name,
                         today)  # the data folder where SAP feature values are stored
outputfile = 'SyllableDuration.csv'

if not os.path.exists(save_path):
    os.mkdir(save_path)

# quit the program if the output file already exists
os.chdir(save_path)
if not os.path.exists(outputfile):
    print('The output file already exists!')
    sys.exit()


def syl_type_(syllable, cluster):
    # function to determine the category of the syllable
    type_str = []
    for syllable in syllable:
        if syllable in cluster.Motif:
            type_str.append('M')  # motif
        elif syllable in cluster.Calls:
            type_str.append('C')  # call
        elif syllable in cluster.IntroNotes:
            type_str.append('I')  # intro notes
        else:
            type_str.append(None)  # intro notes
    return type_str


# Extract syllable duration from EventInfo.m
df = pd.DataFrame()
for cluster_run in range(0, nb_cluster):
    cluster = load.cluster(summary_cluster, cluster_run)

    # print(cluster)
    session_id, cell_id, cell_path = load.cluster_info(cluster)
    print('Accessing... ' + cell_path)
    os.chdir(cell_path)

    # Load EventInfo.mat (data structure)
    # FileInd = 1;
    # FileName = 2;
    # FileStart = 3;
    # SyllableOnsets = 4;
    # SyllableOffsets = 5;
    # FileEnd = 6;
    # SyllableNotes = 7;
    # SongContext = 8;

    mat_file = [file for file in os.listdir(cell_path) if file.endswith('EventInfo.mat')][0]
    mat_file = scipy.io.loadmat(mat_file)['EventInfo']
    # print(mat_file)

    nb_rhd_files = len(mat_file)

    for rhd in mat_file:
        file_nb = rhd[0][0]
        file_id = rhd[1][0]
        print('rhd # {} - {} being processed'.format(file_nb, file_id))
        syllable = list(rhd[6][0])  # list type
        nb_syllable = len(syllable)
        context = list(rhd[7][0])  # list type
        syl_type = syl_type_(syllable, cluster)  # list type
        syl_duration = (rhd[4] - rhd[3])[0]  # in seconds

        # Save results to a dataframe

        temp_df = []
        temp_df = pd.DataFrame({'Key': [cluster.Key] * nb_syllable,
                                'BirdID': [cluster.BirdID] * nb_syllable,
                                'TaskName': [cluster.TaskName] * nb_syllable,
                                'TaskSession': [cluster.TaskSession] * nb_syllable,
                                'TaskSessionDeafening': [cluster.TaskSessionDeafening] * nb_syllable,
                                'TaskSessionPostdeafening': [cluster.TaskSessionPostdeafening] * nb_syllable,
                                'DPH': [cluster.DPH] * nb_syllable,
                                'Block_10days': [cluster.Block_10days] * nb_syllable,
                                'FileID': [file_id] * nb_syllable,
                                'Context': context,
                                'SyllableType': syl_type,
                                'Syllable': syllable,
                                'Duration': syl_duration,
                                })
        df = df.append(temp_df, ignore_index=True)
    # break

    # Todo : save file & plot the results

# Save
os.chdir(save_path)
df.to_csv(outputfile, index=False)  # save the dataframe to .cvs format
subprocess.Popen(save_path)  # open the folder
