"""
By Jaerong
Get PSD similarity to measure changes in song after deafening
"""

from analysis.spike import *
from analysis.song import *
import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pandas as pd
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
fig_save_ok = True
file_save_ok = False
save_psd = True
update = False
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 2"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()  # data path
    channel_nb = int(cluster_db.channel[-2:])
    unit_nb = int(cluster_db.unit[-2:])
    format = cluster_db.format

    if type(channel_nb) == int:
        ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster class object
    else:
        si = SongInfo(path, name, update=update)  # song class object

    save_path = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / name
    psd_array, psd_list, file_list, psd_notes = get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

    # Organize data into a dictionary
    # data = {
    #     'psd_array': psd_array,
    #     'psd_list': psd_list,
    #     'file_list': file_list,
    #     'psd_notes': psd_notes,
    # }

    bird_id = cluster_db.birdID
    task_name = cluster_db.taskName

    if task_name == 'Predeafening':
        npz_name = bird_id + '_' + task_name
        npz_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npz_name

        if npz_name.exists():
            data = np.load(npz_name, allow_pickle=True).item()
            # psd_array, psd_list, file_list, psd_notes = \
            #     data['psd_array'], data['psd_list'], data['file_list'], data['psd_notes']

        else:
            # Save results
            np.save(npz_name, data)


    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', add_date=False)
        # save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)



print('Done!')