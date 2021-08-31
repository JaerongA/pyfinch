# By Jaerong
# Syllable duration analysis

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from summary.read_config import parser
from summary import load

project_path = load.project(parser)  # find the project folder
# summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file

data_path = os.path.join(project_path, r'Analysis\SAP_features')  # the data folder where SAP feature values are stored
analysis_file = 'SAP(ALL).txt'

os.chdir(data_path)
df = pd.read_csv(analysis_file, delimiter="\t")
df = df.query('Context == "Undir"')  # select only Undir trials


def unique(list):
    # Extract unique strings from the list in the order they appeared
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


bird_list = unique(df['BirdID'].tolist())
task_list = unique(df['TaskName'].tolist())

for bird in bird_list:
    for task in task_list:

        note_list = unique(df['Note'].tolist())
        # bird = 'b70r38'
        # task = 'Predeafening'

        temp_df = []
        temp_df = df.loc[(df['BirdID'] == bird) & (df['TaskName'] == task)]
        note_list = unique(temp_df['Note'].tolist())

        title = '-'.join([bird, task])
        fig = plt.figure(figsize=(5, 4))
        plt.suptitle(title, size=10)
        # ax = sns.distplot((temp_df['Duration'], hist= False, kde= True)
        ax = sns.distplot(temp_df['Duration'], hist=False, kde=True)
        temp_df.groupby(['Note'])['Duration'].mean()  # mean duration of each note

        # mark each note
        mean_dur = list(zip(note_list, temp_df.groupby(['Note'])[
            'Duration'].mean().to_list()))  # [('a', 236.3033654971783), ('b', 46.64262295081962), ('c', 154.57333333333335), ('d', 114.20039483457349)]
        for note, dur in mean_dur:
            plt.axvline(dur, color='k', linestyle='dashed', linewidth=1)

        plt.show()
        print('Prcessing... {} from Bird {}'.format(task, bird))

        # ax = sns.scatterplot(x='PCA1', y='PCA2', data=temp_df, hue='Note', size=2)
        # legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        # # legend.texts[-2:] = ''
        # ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
        #
        # # plt.show()
        # # print(title)

# ------------------------------------------------------------------------------------------
