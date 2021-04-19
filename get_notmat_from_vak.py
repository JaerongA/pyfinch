"""
By Jaerong
create .not.mat files based on the output .csv generated by vak
"""

import numpy as np
# import os
import pandas as pd
from pathlib import Path
import scipy.io


def unique(list):
    """Extract unique strings from the list in the order they appeared"""
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


# Save new .not.mat in this folder
save_dir = Path().absolute() / 'prediction'
if not save_dir.exists():
    save_dir.mkdir(parents=True)

# Read from .csv
csv_name = 'y44r34_annotation_results.csv'  # vak prediction output

df = pd.read_csv(csv_name)

file_list = unique(df['audio_file'].tolist())

temp_df = []

for audio_file in file_list:
    temp_df = []
    temp_df = df.loc[(df['audio_file'] == audio_file)]

    print('Processing...  ' + audio_file)
    notmat_file = audio_file + '.not.mat'
    syllables = np.array(''.join(map(str, temp_df.label)))  # syllables
    onsets = np.array(temp_df.onset_s) * 1E3  # syllable onset timestamp
    offsets = np.array(temp_df.offset_s) * 1E3  # syllable offset timestamp

    mdict = {'syllables': syllables,
             'onsets': onsets,
             'offsets': offsets}

    file_name = save_dir / notmat_file
    scipy.io.savemat(file_name, mdict)  # store above values to a new .not.mat