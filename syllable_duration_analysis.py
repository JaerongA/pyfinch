# By Jaerong (06/04/2020)
# Syllable duration for all syllables regardless of it type
# Calculation based on EventInfo.m

import os
from summary.read_config import parser
from summary import load
from summary import save
from datetime import date


project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del parser

path_name = 'SyllableDuration'
today = date.today().strftime("%Y-%m-%d")  # 2020-07-04
save_path = os.path.join(project_path, r'Analysis', path_name, today)  # the data folder where SAP feature values are stored
if not os.path.exists(save_path):
    os.mkdir(save_path)



