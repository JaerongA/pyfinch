"""
Runs PCA analysis on song features generated from SAP (Song Analysis Pro 10)
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from summary.read_config import parser
from summary import load

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file

data_path = os.path.join(project_path, r'Analysis\SAP_features') # the data folder where SAP feature values are stored
analysis_file = 'SAP(ALL).txt'

os.chdir(data_path)
df = pd.read_csv(analysis_file, delimiter="\t")
df = df.query('Context == "Undir"')  # select only Undir trials

# df.columns

df_pca = df.loc[:, 'AmplitudeModulation':'PitchGoodness']

# Preprocessing - standardize the value
scaler = StandardScaler()
scaler.fit(df_pca)
scaled_data = scaler.transform(df_pca)

# Build PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
# pca_data.shape

df['PCA1'] = pca_data[:,0]
df['PCA2'] = pca_data[:,1]
df['PCA3'] = pca_data[:,2]




#----------- PCA section for different conditions -----------------

def unique(list):
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]

bird_list = unique(df['BirdID'].tolist())
task_list = unique(df['TaskName'].tolist())


print(bird_list)


for bird in bird_list:
    for task in task_list:
        print('Prcessing... {} from Bird {}'.format(task, bird))

        temp_df = []
        temp_df = df.loc[(df['BirdID'] == bird) & (df['TaskName'] == task)]

        fig = plt.figure(figsize=(5, 4))
        title = '-'.join([bird, task])
        plt.suptitle(title, size=10)
        ax = sns.scatterplot(x='PCA1', y='PCA2', data=temp_df, hue='Note', size=2)
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        # legend.texts[-2:] = ''
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

        # plt.show()
        # print(title)

#------------------------------------------------------------------------------------------
















# Plot the results
fig = plt.figure(figsize=(18, 9))

# plt.suptitle('PitchGoodness (Undir)', size=20)
circ_size = 1

# Create multiple dataframes per bird
df_b70r38 = df[df['BirdID'] == 'b70r38']
df_g35r38 = df[df['BirdID'] == 'g35r38']
df_w16w14 = df[df['BirdID'] == 'w16w14']
df_b4r64 = df[df['BirdID'] == 'b4r64']
df_b14r74 = df[df['BirdID'] == 'b14r74']
df_w21w30 = df[df['BirdID'] == 'w21w30']
df_g70r40 = df[df['BirdID'] == 'g70r40']

gs = gridspec.GridSpec(2, 9)
ax1a = plt.subplot(gs[0, 1:3])
sns.lineplot(x='TaskSessionPostdeafening', y='PCA1', hue='Note', data=df_b70r38, ci=None, marker='o',
             mew=circ_size)
ax1a.spines['right'].set_visible(False), ax1a.spines['top'].set_visible(False)
ax1a.set_ylabel('PCA1'), ax1a.set_title('b70r38')
ax1a.set_xlabel('')
# ax1a.set_ylim([0.75, 0.95])


ax1b = plt.subplot(gs[0, 3:5], sharey=ax1a, sharex=ax1a)
sns.lineplot(x='TaskSessionPostdeafening', y='PCA1', hue='Note', data=df_g35r38, ci=None, marker='o',
             mew=circ_size)
ax1b.spines['right'].set_visible(False), ax1b.spines['top'].set_visible(False)
ax1b.set_ylabel(''), ax1b.set_title('g35r38')
ax1b.set_xlabel('')
# ax1b.set_ylim([0, 0.035])
fig.tight_layout()

ax1c = plt.subplot(gs[0, 5:7], sharey=ax1a, sharex=ax1a)
sns.lineplot(x='TaskSessionDeafening', y='PCA1', hue='Note', data=df_w16w14, ci=None, marker='o',
             mew=circ_size)
ax1c.spines['right'].set_visible(False), ax1c.spines['top'].set_visible(False)
ax1c.set_ylabel(''), ax1c.set_title('w16w14')
ax1c.set_xlabel('')
# ax1b.set_ylim([0, 0.035])
fig.tight_layout()

ax2a = plt.subplot(gs[1, :2], sharey=ax1a, sharex=ax1a)
sns.lineplot(x='TaskSessionDeafening', y='PCA1', hue='Note', data=df_b14r74, ci=None, marker='o',
             mew=circ_size)
ax2a.spines['right'].set_visible(False), ax2a.spines['top'].set_visible(False)
ax2a.set_ylabel('PCA1'), ax2a.set_title('b14r74')
# ax2a.set_ylim([0.75, 0.95])


ax2b = plt.subplot(gs[1, 2:4], sharey=ax2a, sharex=ax1a)
sns.lineplot(x='TaskSessionDeafening', y='PCA1', hue='Note', data=df_b4r64, ci=None, marker='o',
             mew=circ_size)
ax2b.spines['right'].set_visible(False), ax2b.spines['top'].set_visible(False)
ax2b.set_ylabel(''), ax2b.set_title('b4r64')

ax2c = plt.subplot(gs[1, 4:6], sharey=ax2a, sharex=ax1b)
sns.lineplot(x='TaskSessionDeafening', y='PCA1', hue='Note', data=df_w21w30, ci=None, marker='o',
             mew=circ_size)
ax2c.spines['right'].set_visible(False), ax2c.spines['top'].set_visible(False)
ax2c.set_ylabel(''), ax2c.set_title('w21w30')

ax2c = plt.subplot(gs[1, 6:8], sharey=ax2a, sharex=ax1b)
sns.lineplot(x='TaskSessionDeafening', y='PCA1', hue='Note', data=df_g70r40, ci=None, marker='o',
             mew=circ_size)
ax2c.spines['right'].set_visible(False), ax2c.spines['top'].set_visible(False)
ax2c.set_ylabel(''), ax2c.set_title('g70r40')

fig.tight_layout()

fig.subplots_adjust(top=0.88)
plt.show()
# plt.savefig("PCA1.pdf", transparent=True)
