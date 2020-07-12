"""By Jaerong (05/05/2020)"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from decimal import Decimal

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import warnings

warnings.filterwarnings('ignore')

## Load Data
# projectROOT = r'C:\Users\jahn02\Box\Data\Deafening Project';  # lab
project_path = r'H:\Box\Data\Deafening Project'  # home
analysisROOT = project_path + '\Analysis\Summary'
saveROOT = project_path + '\Analysis'

os.chdir(analysisROOT)
# os.listdir()

analysis_file = "Cluster_summary(Deafening).xlsx"
summary = pd.read_excel(analysis_file, index_col='Key')
summary = summary[summary['AnalysisOK'] == 1]

# print(summary.shape)
# summary
# summary.columns



## Obtain entropy per syllable in one session

plt.figure(figsize=(5, 4))

# ax = sns.stripplot(df["Block"], df["EntropyNorm"],
#                    size=6,  jitter=True,
#                    edgecolor="gray",  alpha =.7)


ax = sns.scatterplot(x="SongFRCrossCorrMaxLoc", y="SongFRCrossCorrR", data=summary, hue = "TaskName",
                    style="BirdID", s = 55, alpha = 0.8)


ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
plt.xlabel('Peak Latency (ms)'), plt.ylabel('CrossCorr')
ax.set_xlim([-110, 110])
ax.set_ylim([0.3, 0.6])
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.axvline(x=0, linestyle = '--', color = 'k', linewidth = 0.5)

# Save results
os.chdir(saveROOT)
plt.savefig("SyllableFR_CrossCorr.pdf", transparent=True, bbox_inches='tight')
