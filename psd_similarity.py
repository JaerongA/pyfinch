"""
By Jaerong
Get PSD similarity to measure changes in song after deafening
"""

from analysis.spike import *
from analysis.song import *
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.pylab import psd
import scipy
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
save_psd = False
update = False
fig_ext = '.png'

# Load database
db = ProjectLoader().load_db()

# Select the birds to use that have both pre and post deafening songs
df = db.to_dataframe("SELECT DISTINCT birdID, taskName FROM cluster")  # convert to dataframe by using pandas lib
bird_list = pd.unique(df.birdID).tolist()
task_list =  pd.unique(df.taskName).tolist()
bird_to_use = []

for bird in bird_list:
    task_list = df.loc[df['birdID'] == bird]['taskName'].to_list()
    if 'Predeafening' in task_list and 'Postdeafening' in task_list:
        bird_to_use.append(bird)

# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id <= 5"
# query = "SELECT * FROM cluster WHERE id = 5"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)

    if not cluster_db.birdID in bird_to_use:  # skip if the bird doesn't have both pre and post deafening songs
        continue

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

    # Check if PSD file already exists
    save_path = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / ci.name  # path to save psd output

    bird_id = cluster_db.birdID
    task_name = cluster_db.taskName
    npy_name = bird_id + '_' + task_name + '.npy'
    npy_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npy_name

    if npy_name.exists():
        data_all = np.load(npy_name, allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

        if ci.name not in data_all['cluster_name']: # append to the existing file

            # Get psd
            # This will create PSD.npy in each cluster folder
            # Note spectrograms & .npy per bird will be stored in PSD_similarity folder
            psd_array, psd_list, file_list, psd_notes = \
                get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

            # Organize data into a dictionary
            data = {
                'psd_list': psd_list,
                'file_list': file_list,
                'psd_notes': psd_notes,
                'cluster_name': [ci.name]
            }

            data_all['psd_list'].extend(data['psd_list'])
            data_all['file_list'].extend(data['file_list'])
            data_all['psd_notes'] += data['psd_notes']
            data_all['cluster_name'].extend(data['cluster_name'])
            np.save(npy_name, data_all)
    else:
        # Get psd
        # This will create PSD.npy in each cluster folder
        # Note spectrograms & .npy per bird will be stored in PSD_similarity folder
        psd_array, psd_list, file_list, psd_notes = \
            get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

        # Organize data into a dictionary
        data = {
            'psd_array': psd_array,
            'psd_list': psd_list,
            'file_list': file_list,
            'psd_notes': psd_notes,
            'cluster_name': ci.name
        }
        np.save(npy_name, data)


    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', add_date=False)
        # save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext, open_folder=True)


# Calculate syllable similarity
bird_list = ['b70r38']

for bird_id in bird_list:

    for task_name in task_list:
        npy_name = bird_id + '_' + task_name + '.npy'
        npy_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npy_name
        data = np.load(npy_name, allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

        #  Load data
        if task_name == 'Predeafening':
            psd_array_pre, notes_pre = data['psd_array'], data['psd_notes']
            psd_list_basis, note_list_basis = get_basis_psd(psd_array_pre, notes_pre)
        elif task_name == 'Postdeafening':
            psd_array_post, notes_post = data['psd_array'], data['psd_notes']

        # Get similarity per syllable
        # Get psd distance
        distance = scipy.spatial.distance.cdist(psd_list_basis, psd_array_post,
                                                'sqeuclidean')  # (number of test notes x number of basis notes)

        # Convert to similarity matrices
        similarity = 1 - (distance / np.max(distance))  # (number of test notes x number of basis notes)

print('Done!')