"""
By Jaerong
create .not.mat files based on the output .csv generated from vak
"""

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io


def unique(list):
    """Extract unique strings from the list in the order they appeared"""
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


def main(csv_path):

    csv_path = Path(csv_path)

    # Save new .not.mat in this folder
    save_dir = csv_path.parent

    # Read from .csv
    df = pd.read_csv(csv_path)
    file_list = unique(df['audio_file'].tolist())

    for audio_file in file_list:

        temp_df = df.loc[(df['audio_file'] == audio_file)]

        print('Processing...  ' + audio_file)
        notmat_file = audio_file + '.not.mat'
        syllables = np.array(''.join(map(str, temp_df.label)))  # syllables
        onsets = np.array(temp_df.onset_s).reshape(-1,1) * 1E3  # syllable onset timestamp
        offsets = np.array(temp_df.offset_s).reshape(-1,1) * 1E3  # syllable offset timestamp
        mdict = {'syllables': syllables,
                 'onsets': onsets,
                 'offsets': offsets}

        file_name = save_dir / notmat_file
        scipy.io.savemat(file_name, mdict)  # store above values to new .not.mat

    print("Done!")


if __name__ == '__main__':

    csv_path = 'H:/Box/Data/BMI/TestSet/results/k71o7_annotation_results.csv'  # vak prediction output

    main(csv_path)
