"""
By Jaerong (2020/06/04)
Syllable duration for all syllables regardless of it type
Calculation based on EventInfo.m
"""


from datetime import date
from math import ceil
import os
import pandas as pd
from summary import load
from summary.read_config import parser
import scipy.io


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

# move to the plotting section if the file already exists
# or generate the file
os.chdir(save_path)
if os.path.exists(outputfile):
    print('The file already exists!')
else:

    def syl_type_(syllable, cluster):
        """ function to determine the category of the syllable """
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
    os.startfile(save_path)  # open the folder

# If the data file already exists, plot the results

df = pd.read_csv(outputfile)
df = df.query('Context == "U"')  # select only Undir trials
df = df.query('Syllable != "0"')  # eliminate non-labeled syllables (e.g., 0)


def unique(list):
    # Extract unique strings from the list in the order they appeared
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


import matplotlib.pyplot as plt
import seaborn as sns

bird_list = unique(df['BirdID'].tolist())
task_list = unique(df['TaskName'].tolist())

for bird in bird_list:

    for task in task_list:

        note_list = unique(df['Syllable'].tolist())

        # bird = 'b70r38'
        # task = 'Predeafening'

        temp_df = []
        temp_df = df.loc[(df['BirdID'] == bird) & (df['TaskName'] == task)]


        note_list = unique(temp_df.query('SyllableType == "M"')['Syllable'])  # only motif syllables

        title = '-'.join([bird, task])
        fig = plt.figure(figsize=(6, 5))
        plt.suptitle(title, size=10)
        # ax = sns.distplot((temp_df['Duration'], hist= False, kde= True)
        # ax = sns.kdeplot(temp_df['Duration'], bw=0.1, label='')
        ax = sns.kdeplot(temp_df['Duration'], bw=0.05, label='', color='k', linewidth=2)
        kde = zip(ax.get_lines()[0].get_data()[0], ax.get_lines()[0].get_data()[1])
        # sns.rugplot(temp_df['Duration'])  # shows ticks
        # temp_df.groupby(['Syllable'])['Duration'].mean()  # mean duration of each note

        import pylab as p

        # https: // stackoverflow.com / questions / 43565892 / python - seaborn - distplot - y - value - corresponding - to - a - given - x - value
        #TODO: extrapolate value and mark with an arrow


        # mark each note
        median_dur = list(zip(note_list, temp_df.query('SyllableType == "M"').groupby(['Syllable'])[
            'Duration'].mean().to_list()))  # [('a', 236.3033654971783), ('b', 46.64262295081962), ('c', 154.57333333333335), ('d', 114.20039483457349)]
        for note, dur in median_dur:
            plt.axvline(dur, color='k', linestyle='dashed', linewidth=1)
            p.arrow(dur, 5, 0, -1)
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
        plt.xlabel('Duration (s)')
        plt.xlim(0, ceil(temp_df['Duration'].max()*10)/10)
        plt.show()


        import IPython
        IPython.embed()



        print('Prcessing... {} from Bird {}'.format(task, bird))


        # plt.savefig(title + '.pdf', transparent=True, bbox_inches='tight')
        plt.savefig(title + '.png', transparent=True)
        break
    break
# a = temp_df[temp_df['Duration'] == temp_df['Duration'].min()]


