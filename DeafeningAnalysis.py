import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
%matplotlib inline

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value



project_path = r'C:\Users\jahn02\Box\Data\Deafening Project';  # lab
# projectROOT =r'C:\Users\AJR\Box\Data\Deafening Project';  # home
analysisROOT = project_path + '\Analysis\Summary'
summary_cluster = "Cluster_summary(Deafening).xlsx"

os.chdir(analysisROOT)
# os.listdir()

summary = pd.read_excel(summary_cluster, index_col='Key')



# summary = pd.read_excel(summary_cluster)
summary_EphysOK = summary[summary['EphysOK']  == 1]

# usable = summary['EphysOK'] != 0
# summary_EphysOK = summary[usable]


## Create multiple dataframes
summary_Predeafening = summary[summary['TaskName'] == 'Predeafening']
summary_Predeafening = summary_Predeafening.reset_index(drop=True)
summary_Postdeafening = summary[summary['TaskName'] == 'Postdeafening']
summary_Postdeafening = summary_Postdeafening.reset_index(drop=True)


# summary.columns
summary.head()


