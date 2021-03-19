"""
By Jaerong
Get PSD similarity to measure changes in song after deafening
"""

from analysis.spike import ClusterInfo
from pathlib import Path

import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.pylab import psd
import random
import scipy
from scipy import spatial
from scipy.io import wavfile
from scipy.stats import sem
import seaborn as sns
from database.load import *
from analysis.functions import *
from util import save
from util.draw import *
from util.functions import *
from util.spect import *
import gc



# Parameters
save_fig = True
save_psd = True
update = False
save_heatmap = True
fig_ext = '.png'
num_note_crit = 10


def get_bird_list(db):
    # Select the birds to use that have both pre and post deafening songs
    df = db.to_dataframe("SELECT DISTINCT birdID, taskName FROM cluster")  # convert to dataframe by using pandas lib
    bird_list = pd.unique(df.birdID).tolist()
    task_list = pd.unique(df.taskName).tolist()
    bird_to_use = []

    for bird in bird_list:
        task_list = df.loc[df['birdID'] == bird]['taskName'].to_list()
        if 'Predeafening' in task_list and 'Postdeafening' in task_list:
            bird_to_use.append(bird)
    return bird_list, task_list, bird_to_use


def get_psd_bird(db, *bird_list):

    # Make PSD.npy file for each cluster
    # Cluster PSD.npy will be concatenated per bird in a separate folder
    if bird_list:
        for birdID in bird_list:
            query = "SELECT * FROM cluster WHERE birdID == birdID"
        else:
            query = "SELECT * FROM cluster"
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
                    psd_list, file_list, psd_notes = \
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
                psd_list, file_list, psd_notes = \
                    get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

                # Organize data into a dictionary
                data = {
                    'psd_list': psd_list,
                    'file_list': file_list,
                    'psd_notes': psd_notes,
                    'cluster_name': [ci.name]
                }
                np.save(npy_name, data)

def psd_split(psd_list_pre_all, notes_pre_all):
    """
    Randomize the pre data and split into half.
    one half will be used as a basis and the other for getting control similarity
    """
    np.random.seed(0)
    psd_arr_pre_all = np.asarray(psd_list_pre_all)

    arr = np.arange(psd_arr_pre_all.shape[0])
    np.random.shuffle(arr)  # randomize the array
    psd_arr_pre_1st = psd_arr_pre_all[arr[:int(arr.shape[0] / 2)], :]
    psd_arr_pre_2nd = psd_arr_pre_all[arr[int(arr.shape[0] / 2):], :]

    psd_list_1st = []
    psd_list_2nd = []

    for row in psd_arr_pre_1st:
        psd_list_1st.append(row)

    for row in psd_arr_pre_2nd:
        psd_list_2nd.append(row)

    notes_pre_1st = ''  # will be used as basis
    for ind in arr[:int(arr.shape[0] / 2)]:
        notes_pre_1st += notes_pre_all[ind]

    notes_pre_2nd = ''  # will be used as control
    for ind in arr[int(arr.shape[0] / 2):]:
        notes_pre_2nd += notes_pre_all[ind]

    return psd_list_1st, psd_list_2nd, notes_pre_1st, notes_pre_2nd


def get_similarity_heatmap(psd_list_target, psd_list_basis, notes_target, notes_basis,
                           save_heatmap=True):
    """
    Get similarity per syllable
    Parameters
    ----------
    psd_list_target : list
        list of target psd (pre or post-deafening)
    psd_list_basis : list
        list of basis psd (random selection from pre-deafening)
    notes_target : str
        target notes
    notes_basis : str
        basis notes
    Returns
    -------
    """

    # Get psd distance
    distance = \
        scipy.spatial.distance.cdist(psd_list_target, psd_list_basis, 'sqeuclidean')  # (number of test notes x number of basis notes)

    # Plot similarity matrix per syllable
    notes_list_basis = unique(notes_basis)  # convert syllable string into a list of unique syllables
    notes_list_target = unique(notes_target)

    # Get similarity matrix per test note
    for note in notes_list_target:  # loop through notes

        if note not in notes_list_basis:  # skip if the note doesn't exist in the basis set
            continue

        ind = find_str(notes_target, note)
        nb_note = len(ind)
        if nb_note < num_note_crit:
            continue

        # Get distance matrix per note
        note_distance = distance[ind, :]

        # Convert to similarity matrices
        note_similarity = 1 - (note_distance / np.max(note_distance))  # (number of test notes x number of basis notes)

        # Get mean or median similarity index
        similarity_mean = np.expand_dims(np.mean(note_similarity, axis=0), axis=0)  # or axis=1
        similarity_sem = sem(note_similarity, ddof=1)
        # similarity_median = np.expand_dims(np.median(note_similarity, axis=0), axis=0)  # or axis=1

        # Plot the similarity matrix
        fig = plt.figure(figsize=(5, 5))
        # title = "Sim matrix: note = {}".format(note)
        fig_name = f"note - {note}"
        title = f"{bird_id} Sim matrix: note = {note} ({nb_note})"
        gs = gridspec.GridSpec(7, 8)
        ax = plt.subplot(gs[0:5, 1:7])
        ax = sns.heatmap(note_similarity,
                         vmin=0, vmax=1,
                         cmap='binary')
        ax.set_title(title)
        ax.set_ylabel('Test syllables')
        ax.set_xticklabels(note_list_basis)
        plt.tick_params(left=False)
        plt.yticks([0.5, nb_note - 0.5], ['1', str(nb_note)])

        ax = plt.subplot(gs[-1, 1:7], sharex=ax)

        ax = sns.heatmap(similarity_mean, annot=True, cmap='binary',
                         vmin=0, vmax=1,
                         annot_kws={"fontsize": 7})
        ax.set_xlabel('Basis syllables')
        ax.set_yticks([])
        ax.set_xticklabels(note_list_basis)
        # plt.show()

        similarity_mean_val = similarity_mean[0][note_list_basis.index(note)]
        # similarity_median_val = similarity_median[0][note_list_basis.index(note)]

        # Save heatmap (similarity matrix)
        if save_heatmap:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'PSD_similarity' + '/' + bird_id, add_date=False)
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=False)
        else:
            plt.close(fig)


# Load database
db = ProjectLoader().load_db()
bird_list, task_list, bird_to_use = get_bird_list(db)  # get bird list to analyze from db

# Calculate PSD similarity
bird_to_use = ['b70r38']
for bird_id in bird_to_use:

    for task_name in task_list:
        npy_name = bird_id + '_' + task_name + '.npy'
        npy_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npy_name
        data = np.load(npy_name, allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

        #  Load data
        if task_name == 'Predeafening':
            psd_list_pre_all, notes_pre_all = data['psd_list'], data['psd_notes']

            # Split pre-deafening PSDSs into halves
            # 1st half will be used as basis and the 2nd half as control
            psd_list_basis, psd_list_pre, notes_basis, notes_pre = psd_split(psd_list_pre_all, notes_pre_all)
            del psd_list_pre_all, notes_pre_all

            # Get basis psd and list of basis syllables
            psd_list_basis, note_list_basis = get_basis_psd(psd_list_basis, notes_basis)
            # Get similarity heatmap between basis and pre-deafening control
            get_similarity_heatmap(psd_list_pre, psd_list_basis, notes_pre, notes_basis,
                                   save_heatmap=True)

        elif task_name == 'Postdeafening':
            psd_list_post, notes_post = data['psd_list'], data['psd_notes']
            # Get similarity heatmap between basis and post-deafening
            get_similarity_heatmap(psd_list_post, psd_list_basis, notes_post, notes_basis,
                                   save_heatmap=True)



gc.collect()
print('Done!')

