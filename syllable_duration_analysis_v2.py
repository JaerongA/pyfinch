"""
By Jaerong (2020/06/04)
Syllable duration for all syllables regardless of it type
Calculation based on EventInfo.m
"""

from database import load
from datetime import date
from math import ceil
import numpy as np
from pathlib import Path
from song.functions import *
import pandas as pd
# from summary import load
# from summary.read_config import parser
import scipy.io
from utilities import save

# Create save dir
dir_name = 'SyllableDuration'
save_dir = save.make_save_dir(dir_name)

# Store results in the dataframe
df = pd.DataFrame()

# Load song database
# query = "SELECT * FROM song"
query = "SELECT * FROM song WHERE id = 3"

cur, conn, col_names = load.database(query)

for song_info in cur.fetchall():
    song_name, song_path = load.song_info(song_info)
    print('Accessing... ' + song_name)

    # Store values in a list
    duration_list = list()
    syllable_list = list()
    context_list = list()
    bout_list = list()

    for site in [x for x in song_path.iterdir() if x.is_dir()]:  # loop through different sites on the same day

        mat_files = [file for file in site.rglob('*.not.mat')]

        for file in mat_files:
            # Load variables from .not.mat
            print(file.name)
            # print(file)
            onsets, offsets, intervals, duration, syllables, context = \
                read_not_mat(file)

            # Store values in a list
            context_list.append(
                file.name.split('.')[0].split('_')[-1][0].upper())  # extract 'U' or 'D' from the file name
            duration_list.append(duration)
            syllable_list.append(syllables)
            nb_syllable = len(syllables)

            # Save results to a dataframe
            temp_df = []
            temp_df = pd.DataFrame({'SongID': song_info['id'] * nb_syllable,
                                    'BirdID': [song_info.BirdID] * nb_syllable,
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





    import IPython
    IPython.embed()
    break

    #
    #
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # bird_list = unique(df['BirdID'].tolist())
    # task_list = unique(df['TaskName'].tolist())
    #
    # for bird in bird_list:
    #
    #     for task in task_list:
    #
    #         note_list = unique(df['Syllable'].tolist())
    #
    #         # bird = 'b70r38'
    #         # task = 'Predeafening'
    #
    #         temp_df = []
    #         temp_df = df.loc[(df['BirdID'] == bird) & (df['TaskName'] == task)]
    #
    #         note_list = unique(temp_df.query('SyllableType == "M"')['Syllable'])  # only motif syllables
    #
    #         title = '-'.join([bird, task])
    #         fig = plt.figure(figsize=(6, 5))
    #         plt.suptitle(title, size=10)
    #         # ax = sns.distplot((temp_df['Duration'], hist= False, kde= True)
    #         # ax = sns.kdeplot(temp_df['Duration'], bw=0.1, label='')
    #         ax = sns.kdeplot(temp_df['Duration'], bw=0.05, label='', color='k', linewidth=2)
    #         kde = zip(ax.get_lines()[0].get_data()[0], ax.get_lines()[0].get_data()[1])
    #         # sns.rugplot(temp_df['Duration'])  # shows ticks
    #         # temp_df.groupby(['Syllable'])['Duration'].mean()  # mean duration of each note
    #
    #         import pylab as p
    #
    #         # https: // stackoverflow.com / questions / 43565892 / python - seaborn - distplot - y - value - corresponding - to - a - given - x - value
    #         # TODO: extrapolate value and mark with an arrow
    #
    #         # mark each note
    #         median_dur = list(zip(note_list, temp_df.query('SyllableType == "M"').groupby(['Syllable'])[
    #             'Duration'].mean().to_list()))  # [('a', 236.3033654971783), ('b', 46.64262295081962), ('c', 154.57333333333335), ('d', 114.20039483457349)]
    #         for note, dur in median_dur:
    #             plt.axvline(dur, color='k', linestyle='dashed', linewidth=1)
    #             p.arrow(dur, 5, 0, -1)
    #         ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    #         plt.xlabel('Duration (s)')
    #         plt.xlim(0, ceil(temp_df['Duration'].max() * 10) / 10)
    #         plt.show()
    #
    #         import IPython
    #
    #         IPython.embed()
    #
    #         print('Prcessing... {} from Bird {}'.format(task, bird))
    #
    #         # plt.savefig(title + '.pdf', transparent=True, bbox_inches='tight')
    #         plt.savefig(title + '.png', transparent=True)
    #         break
    #     break
    # # a = temp_df[temp_df['Duration'] == temp_df['Duration'].min()]
