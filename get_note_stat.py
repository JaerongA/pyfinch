import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

from song.analysis import read_not_mat
from utilities.functions import find_str

data_path = Path("H:\Box\Data\BMI\TrainingSet")

audio_files = list(data_path.glob('*.wav'))

# Store values in these lists
duration_array = np.array([], dtype=np.float64)
syllable_list = []

# Store results in the dataframe
df = pd.DataFrame()


for file in audio_files:
    # Load audio files
    print('Loading... ' + file.stem)

    # Load the .not.mat file
    notmat_file = file.with_suffix('.wav.not.mat')
    onsets, offsets, intervals, durations, syllables, context = read_not_mat(notmat_file)

    temp_df = pd.DataFrame({'SongID': [song_info['id']] * nb_syllable,
                            'BirdID': [song_info['birdID']] * nb_syllable,
                            'TaskName': [song_info['taskName']] * nb_syllable,
                            'TaskSession': [song_info['taskSession']] * nb_syllable,
                            'TaskSessionDeafening': [song_info['taskSessionDeafening']] * nb_syllable,
                            'TaskSessionPostdeafening': [song_info[
                                                             'taskSessionPostDeafening']] * nb_syllable,
                            'DPH': [song_info['dph']] * nb_syllable,
                            'Block10days': [song_info['block10days']] * nb_syllable,
                            'FileID': [file.name] * nb_syllable,
                            'Context': [context] * nb_syllable,
                            'SyllableType': syl_type,
                            'Syllable': list(syllables),
                            'Duration': duration,
                            })
    df = df.append(temp_df, ignore_index=True)


    duration_array = np.append(duration_array, durations)
    syllable_list.append(syllables)


syllable_array = ''.join(syllable_list)

for note in set(syllable_array): # loop through unique syllables

    print(note)
    ind = find_str(note, syllable_array)
    print(duration_array[ind])
    break