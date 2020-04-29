##  Created by Jaerong (01/27/2020)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# projectROOT = r'C:\Users\jahn02\Box\Data\Deafening Project';  # lab
projectROOT = r'C:\Users\AJR\Box\Data\Deafening Project'  # home
analysisROOT = projectROOT + '\Analysis\Entropy\Entropy'
# analysis_file = 'Entropy.csv'
analysis_file = 'Entropy(ALL).txt'

os.chdir(analysisROOT)
os.listdir()

df = pd.read_csv(analysis_file, delimiter="\t")


## Treat the column with nans as numeric values
# df.replace({'NaN': np.nan}, regex=True, inplace= True)

# df['FF_CV(Dir)'] = df['FF_CV(Dir)'].astype('Float64')


## Disregard if the number of notes is small

# note_crit = 10

# df.loc[df['Nb_Notes(Undir)'] < note_crit, 'FF_CV(Undir)'] = np.nan
# df.loc[df['Nb_Notes(Dir)'] < note_crit, 'FF_CV(Dir)'] = np.nan


## Disregard if the number of notes is small




# print(df['BirdID'].unique())

# print(df.columns)

# # df.head()
df
