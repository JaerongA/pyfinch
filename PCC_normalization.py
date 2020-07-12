import os
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats
#
# import matplotlib
#
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# from decimal import Decimal


# projectROOT = r'C:\Users\jahn02\Box\Data\Deafening Project';  # lab
project_path = r'C:\Users\AJR\Box\Data\Deafening Project'  # home
analysisROOT = project_path + '\Analysis\Summary'
summary_cluster = "Cluster_summary(Deafening).xlsx"
print(summary_cluster)
os.chdir(analysisROOT)
os.listdir()


summary = pd.read_excel(summary_cluster)
# print(summary)
 # summary = pd.read_excel(summary_cluster, index_col='key')

# ## select only the row to be analyzed
#summary = summary[summary['nb_motifs_undir'] >= 10]
#summary = summary[summary['unitcategory_undir'] == 'bursting']
#
# ## mark predeafening session as 0
#summary.loc[summary['taskname'] == 'predeafening', 'tasksession'] = 0
#
#
#summary_ephysok = summary[summary['ephysok']  == 1]
#
#summary_ephysok['pairwisecorr_undir'] = summary_ephysok['pairwisecorr_undir'].apply(pd.to_numeric)
#
# # summary_ephysok = summary_ephysok[summary_ephysok['nb_motifs_undir'] >= 10]
#
# # usable = summary['ephysok'] != 0
#summary_ephysok = summary[usable]
#
#
# ## create multiple dataframes
##summary_predeafening = summary[summary['taskname'] == 'predeafening']
##summary_predeafening = summary_predeafening.reset_index(drop=True)
##summary_postdeafening = summary[summary['taskname'] == 'postdeafening']
##summary_postdeafening = summary_postdeafening.reset_index(drop=True)
#
#
# # summary.columns
#print(summary)
 # summary.tail()


