# Created by JR (06/02/2020)
# This program reads from SAP(ALL).txt and run analysis on song features (e.g., entropy, gravity center, etc).

import os
from summary import load
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

config_file = 'project.ini'
parser = load.config(config_file)
projectROOT = load.project(parser)  # find the project folder
del config_file, parser

dataROOT = projectROOT + '\\Analysis\\SAP_features'
analysis_file = 'SAP(ALL).txt'
os.chdir(dataROOT)

# Load the data
df = pd.read_csv(analysis_file, delimiter="\t")
df = df.query('Context == "Undir"')  # select only Undir trials
print(df.columns)











# import matplotlib.gridspec as gridspec
#
# fig = plt.figure(figsize=(18,9))
#
# plt.suptitle('PitchGoodness (Undir)', size = 20)
# circ_size = 1
#
#
# ## Create multiple dataframes per bird
#
#
# df_b70r38 = df[df['BirdID'] == 'b70r38']
# df_g35r38 = df[df['BirdID'] == 'g35r38']
# df_w16w14 = df[df['BirdID'] == 'w16w14']
# df_b4r64 = df[df['BirdID'] == 'b4r64']
# df_b14r74 = df[df['BirdID'] == 'b14r74']
# df_w21w30 = df[df['BirdID'] == 'w21w30']
# df_g70r40 = df[df['BirdID'] == 'g70r40']
#
#
#
#
#
# gs = gridspec.GridSpec(2, 9)
# ax1a = plt.subplot(gs[0, 1:3])
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_b70r38, ci = None, marker = 'o', mew=circ_size)
# ax1a.spines['right'].set_visible(False), ax1a.spines['top'].set_visible(False)
# ax1a.set_ylabel('FrequencyModulation'), ax1a.set_title('b70r38')
# ax1a.set_xlabel('')
# # ax1a.set_ylim([0.75, 0.95])
#
#
#
# ax1b = plt.subplot(gs[0, 3:5], sharey= ax1a, sharex= ax1a)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_g35r38, ci = None, marker = 'o', mew=circ_size)
# ax1b.spines['right'].set_visible(False), ax1b.spines['top'].set_visible(False)
# ax1b.set_ylabel(''), ax1b.set_title('g35r38')
# ax1b.set_xlabel('')
# # ax1b.set_ylim([0, 0.035])
# fig.tight_layout()
#
#
# ax1c = plt.subplot(gs[0, 5:7], sharey= ax1a, sharex= ax1a)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_w16w14, ci = None, marker = 'o', mew=circ_size)
# ax1c.spines['right'].set_visible(False), ax1c.spines['top'].set_visible(False)
# ax1c.set_ylabel(''), ax1c.set_title('w16w14')
# ax1c.set_xlabel('')
# # ax1b.set_ylim([0, 0.035])
# fig.tight_layout()
#
#
#
# ax2a = plt.subplot(gs[1, :2], sharey= ax1a, sharex= ax1a)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_b14r74, ci = None, marker = 'o', mew=circ_size)
# ax2a.spines['right'].set_visible(False), ax2a.spines['top'].set_visible(False)
# ax2a.set_ylabel('AmplitudeModulation'), ax2a.set_title('b14r74')
# # ax2a.set_ylim([0.75, 0.95])
#
#
#
# ax2b = plt.subplot(gs[1, 2:4], sharey= ax2a, sharex= ax1a)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_b4r64, ci = None, marker = 'o', mew=circ_size)
# ax2b.spines['right'].set_visible(False), ax2b.spines['top'].set_visible(False)
# ax2b.set_ylabel(''), ax2b.set_title('b4r64')
#
#
#
#
# ax2c = plt.subplot(gs[1, 4:6], sharey= ax2a, sharex= ax1b)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_w21w30, ci = None, marker = 'o', mew=circ_size)
# ax2c.spines['right'].set_visible(False), ax2c.spines['top'].set_visible(False)
# ax2c.set_ylabel(''), ax2c.set_title('w21w30')
#
#
#
# ax2c = plt.subplot(gs[1, 6:8], sharey= ax2a, sharex= ax1b)
# sns.lineplot(x = 'TaskSessionPostdeafening', y = 'PitchGoodness', hue = 'Note', data = df_g70r40, ci = None, marker = 'o', mew=circ_size)
# ax2c.spines['right'].set_visible(False), ax2c.spines['top'].set_visible(False)
# ax2c.set_ylabel(''), ax2c.set_title('g70r40')
#
#
# fig.tight_layout()
#
# fig.subplots_adjust(top=0.88)
#
# plt.savefig("PitchGoodness.pdf", transparent=True)
#
