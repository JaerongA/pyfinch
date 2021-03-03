"""
By Jaerong
Get PSD similarity to measure changes in song after deafening
"""

from analysis.spike import *
import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.pylab import psd
from scipy import spatial
from scipy.io import wavfile
from scipy.stats import sem
from database.load import *
from analysis.functions import *
from analysis.parameters import *
from util import save
from util.draw import *
from util.functions import *
from util.spect import *


# Parameters
save_fig = False
dir_name = 'PSD_similarity'
fig_save_ok = True
file_save_ok = False
save_psd = True
update = False

# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 6"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    audio = AudioData(path, update=update)










    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', dir_name)
        # save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)



print('Done!')