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
save_fig = True
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
# query = "SELECT * FROM cluster WHERE id <= 5"
query = "SELECT * FROM cluster WHERE id = 5"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster()  # data path

    try:
        channel_nb = int(cluster_db.channel[-2:])
    except:
        channel_nb = ''
    try:
        unit_nb = int(cluster_db.unit[-2:])
    except:
        unit_nb = ''
    format = cluster_db.format

    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster class object

    save_path = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / ci.name  # path to save psd output

    # Get psd
    # This will create PSD.npy in each cluster folder
    # Note spectrograms & .npy per bird will be stored in PSD_similarity folder
    psd_array, psd_list, file_list, psd_notes = \
        get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

    # Organize data into a dictionary
    data = {
        'psd_array': psd_array,
        'psd_list': [psd_list],
        'file_list': [file_list],
        'psd_notes' : [psd_notes],
        'cluster_name' : [ci.name]
    }

    bird_id = cluster_db.birdID
    task_name = cluster_db.taskName
    npy_name = bird_id + '_' + task_name + '.npy'
    npy_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npy_name


    if npy_name.exists():
        data_all = np.load(npy_name, allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

        if data['cluster_name'][0] not in data_all['cluster_name']: # append to the existing file
            data_all['psd_array'] = np.append(data_all['psd_array'], data['psd_array'])
            data_all['psd_list'].append(data['psd_list'][0])
            data_all['file_list'].append(data['file_list'][0])
            data_all['psd_notes'].append(data['psd_notes'][0])
            data_all['cluster_name'].append(data['cluster_name'][0])
            np.save(npy_name, data_all)
    else:
        # Save results
        np.save(npy_name, data)


    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', add_date=False)
        # save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)



print('Done!')