"""
By Jaerong
Plot syllable durations to view distributions & detect outliers
This should done after each syllable segmentation with uisonganal.m
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from analysis.functions import read_not_mat
from util.draw import *
from util.functions import *
import math

# Store results in the dataframe
data_path = Path(r"C:\Users\dreill03\Box\AreaX\p84w29\DougPreProcess\DKR\20201208\02\Songs3\Songs4\To Analyze")
audio_files = list(data_path.glob('*.wav'))

df = pd.DataFrame()

for file in audio_files:
    # Load the .not.mat file
    # print('Loading... ' + file.stem)
    notmat_file = file.with_suffix('.wav.not.mat')
    birdID = file.name.split('_')[0]
    onsets, offsets, intervals, durations, syllables, context = read_not_mat(notmat_file)

    nb_syllable = len(syllables)
    temp_df = pd.DataFrame({'FileID': [notmat_file] * nb_syllable,
                            'Syllable': list(syllables),
                            'Duration': durations,
                            })
    df = df.append(temp_df, ignore_index=True)

# Plot the results
syllable_list = sorted(list(set(df['Syllable'].to_list())))

fig, ax = plt.subplots(figsize=(6, 5))
plt.title("{} - {}".format(birdID, data_path.name))
sns.stripplot(ax=ax, x='Syllable', y='Duration', order=syllable_list, s=4, jitter=0.15, data=df)

for syllable, x_loc in zip(syllable_list, ax.get_xticks()):
    nb_syllable = df[df['Syllable'] == syllable]['Syllable'].count()
    max_dur = df[df['Syllable'] == syllable]['Duration'].max()
    text = "({})".format(nb_syllable)
    x_loc -= ax.get_xticks()[-1] * 0.03
    y_loc = max_dur + ax.get_ylim()[1] * 0.05
    plt.text(x_loc, y_loc, text)

ax.set_ylim([0, myround(math.ceil(ax.get_ylim()[1]), base=100)])
remove_right_top(ax)
plt.ylabel('Duration (ms)')
fig.tight_layout()

plt.show()
