"""
By Jaerong
Plot syllable durations to view distributions & detect outliers
This should done after each syllable segmentation with uisonganal.m
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from song.analysis import read_not_mat


# Store results in the dataframe
df = pd.DataFrame()
data_path = Path("H:\Box\Data\BMI\TestSet")
audio_files = list(data_path.glob('*.wav'))

for file in audio_files:
    # Load audio files
    print('Loading... ' + file.stem)

    # Load the .not.mat file
    notmat_file = file.with_suffix('.wav.not.mat')
    onsets, offsets, intervals, durations, syllables, context = read_not_mat(notmat_file)

    nb_syllable = len(syllables)

    temp_df = pd.DataFrame({'FileID': [notmat_file] * nb_syllable,
                            'Syllable': list(syllables),
                            'Duration': durations,
                            })
    df = df.append(temp_df, ignore_index=True)

# Plot the results
syllable_list = sorted(list(set(df['Syllable'].to_list())))

sns.catplot(x='Syllable', y= 'Duration', order=syllable_list, s=4, jitter=0.15, data=df)
plt.ylabel('Duration (ms)')
plt.show()

