"""
Get PSD similarity to measure changes in song after deafening
"""

from analysis.spike import ClusterInfo
from analysis.functions import get_pre_motor_spk_per_note
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.pylab import psd
import random
import scipy
from scipy import spatial
from scipy.io import wavfile
from scipy.stats import sem, pearsonr
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
psd_update = True
update_db = True  # save results to DB
save_heatmap = True  # heatmap & csv
context_selection = 'U'  # use undirected song only (default = None)
fig_ext = '.png'
num_note_crit = 10  # number of notes criteria (only use notes that is greater than or equal to this criteria)
nb_row = 6
nb_col = 6
font_size = 12
alpha = 0.05


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


def get_psd_bird(db, *bird_to_use):
    # Make PSD.npy file for each cluster
    # Cluster PSD.npy will be concatenated per bird in a separate folder
    if bird_to_use:
        # query = "select * from cluster where birdID in {}".format(tuple(bird_to_use[0]))
        # query = "select * from cluster where birdID in {}".format(tuple(bird_to_use[0]))
        query = "SELECT * FROM cluster WHERE birdID IN (?)"
        db.cur.execute(query, tuple(bird_to_use[0]))
    else:
        query = "SELECT * FROM cluster"
        db.cur.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()  # data path

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
            data_all = np.load(npy_name,
                               allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

            if ci.name not in data_all['cluster_name']:  # append to the existing file
                # Get psd
                # This will create PSD.npy in each cluster folder
                # Note spectrograms & .npy per bird will be stored in PSD_similarity folder
                psd_list, file_list, psd_notes, psd_context_list = \
                    get_psd_mat(path, save_path, save_psd=save_psd, update=True, fig_ext=fig_ext)

                # Organize data into a dictionary
                data = {
                    'psd_list': psd_list,
                    'file_list': file_list,
                    'psd_notes': psd_notes,
                    'psd_context_list': psd_context_list,
                    'cluster_name': [ci.name] * len(psd_notes)
                }
                # Concatenate the data
                data_all['psd_list'].extend(data['psd_list'])
                data_all['file_list'].extend(data['file_list'])
                data_all['psd_notes'] += data['psd_notes']
                data_all['psd_context_list'].extend(data['psd_context_list'])
                data_all['cluster_name'].extend(data['cluster_name'])
                np.save(npy_name, data_all)
        else:
            # Get psd
            # This will create PSD.npy in each cluster folder
            # Note spectrograms & .npy per bird will be stored in PSD_similarity folder
            psd_list, file_list, psd_notes, psd_context_list = get_psd_mat(path, save_path, save_psd=save_psd,
                                                                           update=True, fig_ext=fig_ext)

            # Organize data into a dictionary
            data = {
                'psd_list': psd_list,
                'file_list': file_list,
                'psd_notes': psd_notes,
                'psd_context_list': psd_context_list,
                'cluster_name': [ci.name] * len(psd_notes)
            }
            np.save(npy_name, data)
    # print('Done!')


def psd_split(psd_list_pre_all, notes_pre_all, contexts_pre_all):
    """
    Randomize the pre data and split into half.
    one half will be used as a basis and the other for getting control similarity
    """
    import numpy as np

    psd_arr_pre_all = np.asarray(psd_list_pre_all)
    arr = np.arange(psd_arr_pre_all.shape[0])
    np.random.seed(0)
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


def get_similarity_heatmap(psd_list_target, psd_list_basis, notes_target, notes_basis, note_list_basis,
                           file_list=None, cluster_list=None, save_results=True,
                           ):
    """
    Get similarity per syllable
    Parameters
    ----------
    psd_list_target : list
        list of target psd (pre or post-deafening)
        len(psd_list_target) -> number of notes
        psd_list_target[0].shape -> psd vector
    psd_list_basis : list
        list of basis psd (random selection from pre-deafening)
    notes_target : str
        target notes (pre or post-deafening)
    note_basis : str
        basis notes (from pre-deafening)
    note_list_basis : list
        list of notes used as basis note
    file_list : list
        list of files that contain psd
    cluster_list : list
        list of cluster names
    save_results : bool
        save heatmap & csv

    Returns
    -------
    """
    import numpy as np
    # Get psd distance between target and basis
    distance = \
        scipy.spatial.distance.cdist(psd_list_target, psd_list_basis,
                                     'sqeuclidean')  # (number of test notes x number of basis notes)

    # Get basis note info
    nb_note_basis = {}
    for note in note_list_basis:
        ind = find_str(notes_basis, note)
        if len(ind) >= num_note_crit:  # number should exceed the  criteria
            nb_note_basis[note] = len(ind)  # Get the number of basis notes

    notes_list_target = unique(''.join(sorted(notes_target)))

    # Get similarity matrix per test note
    # Store results in the dataframe
    df = pd.DataFrame()
    similarity_info = {}

    for note in notes_list_target:  # loop through notes

        if note not in note_list_basis:  # skip if the note doesn't exist in the basis set
            continue

        ind = find_str(notes_target, note)
        nb_note = len(ind)
        if nb_note < num_note_crit: continue

        # Get distance matrix per note
        note_distance = distance[ind, :]

        # Convert to similarity matrices
        note_similarity = 1 - (note_distance / np.max(note_distance))  # (number of test notes x number of basis notes)

        # Get mean or median similarity index
        similarity_mean = np.expand_dims(np.mean(note_similarity, axis=0), axis=0)  # or axis=1
        similarity_sem = sem(note_similarity, ddof=1)
        # similarity_median = np.expand_dims(np.median(note_similarity, axis=0), axis=0)  # or axis=1

        # Get task condition
        condition = ''
        if task_name == 'Predeafening':  # basis vs. predeafening
            condition = 'Control'
        else:  # basis vs. postdeafening
            condition = 'Deafening'

        # This is to mark the date
        if condition == 'Deafening':
            file_array = np.asarray(file_list)
            note_file = file_array[ind]
            cluster_array = np.asarray(cluster_list)
            note_cluster = cluster_array[ind]

            date_list = []
            for file in note_file:
                date_list.append(file.split('_')[1])
            date_list_unique = unique(date_list)

            # Get date change info
            prev_date = ''
            date_change_ind = []

            for ind, file in enumerate(note_file):
                date = file.split('_')[1]
                if prev_date and prev_date != date:
                    date_change_ind.append(ind - 0.5)
                prev_date = date

        # Plot the similarity matrix
        fig = plt.figure(figsize=(5, 5))
        # title = "Sim matrix: note = {}".format(note)
        fig_name = f"{bird_id}-{condition}_note({note})"
        title = f"{bird_id}-{condition}-Sim matrix: note '{note}' (basis n = {nb_note_basis[note]})"
        gs = gridspec.GridSpec(7, 8)
        ax = plt.subplot(gs[0:5, 1:7])
        ax = sns.heatmap(note_similarity,
                         vmin=0, vmax=1,
                         cmap='binary')

        # Mark change in date
        if 'date_change_ind' in locals():
            for ind in date_change_ind:
                ax.axhline(y=ind, color='r', ls=':', lw=1)

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
        if save_results:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'PSD_similarity' + '/' + bird_id,
                                      add_date=False)
            save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=False)

            # Save results to a dataframe
            # All notes
            temp_df = []
            temp_df = pd.DataFrame({'BirdID': bird_id,
                                    'Condition': task_name,
                                    'Note': note,  # testing note
                                    'NbNotes': [nb_note],
                                    'SimilarityMean': [similarity_mean_val],
                                    })
            df = df.append(temp_df, ignore_index=True)
            csv_path = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / f'{bird_id}_{task_name}.csv'
            temp_df.to_csv(csv_path, index=True, header=True)  # save the dataframe to .cvs format
        else:
            plt.close(fig)

        # Return similarity info only in deafening condition
        if condition == 'Deafening':
            similarity_info[note] = {}
            similarity_info[note]['similarity'] = note_similarity
            similarity_info[note]['note_file'] = note_file
            similarity_info[note]['note_cluster'] = note_cluster

    return similarity_info


def select_context(data, context_selection=None):
    """
    Select data from only one context
    Parameters
    ----------
    data : dict
    context_selection : 'U' or 'D' (None by default)

    Returns
    -------
    data : dict
    """

    if context_selection:  # if not None
        psd_arr = np.asarray(data['psd_list'])
        file_array = np.array(data['file_list'])
        contexts_arr = np.array(data['psd_context_list'])
        cluster_name_arr = np.array(data['cluster_name'])

        ind = np.where(contexts_arr == context_selection)[0]
        psd_arr = psd_arr[ind]
        psd_list = []
        for row in psd_arr:
            psd_list.append(row)

        notes_new = ''
        for i in ind:
            notes_new += data['psd_notes'][i]

        data['psd_list'] = psd_list
        data['psd_notes'] = notes_new
        data['file_list'] = file_array[ind].tolist()
        data['psd_context_list'] = contexts_arr[ind].tolist()
        data['cluster_name'] = cluster_name_arr[ind].tolist()
        return data

def get_similarity_index(db, *bird_to_use):
    pass

# Get similarity matrix per bird
# Load database
db = ProjectLoader().load_db()
bird_list, task_list, bird_to_use = get_bird_list(db)  # get bird list to analyze from db

bird_to_use = ['k27o36']
# bird_to_use = ['b70r38', 'w16w14', 'b4r64', 'b14r74', 'w21w30', 'y44r34', 'k27o36', 'o25w75']

# Make PSD.npy file for each cluster
# Make sure to delete the existing .npy before making a new one
# get_psd_bird(db, bird_to_use)

# Calculate PSD similarity
for bird_id in bird_to_use:
    print('Processing ' + bird_id)
    for task_name in task_list:
        npy_name = bird_id + '_' + task_name + '.npy'
        npy_name = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / npy_name
        data = np.load(npy_name,
                       allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template

        # Select data from one social context ('U' or 'D')
        if context_selection:  # if not None
            data = select_context(data, context_selection=context_selection)

        if not data['psd_list']: continue  # skip if no data exists

        #  Load data
        if task_name == 'Predeafening':
            psd_list_pre_all, notes_pre, file_list_pre, context_list_pre, cluster_name_pre = \
                data['psd_list'], data['psd_notes'], data['file_list'], data['psd_context_list'], data['cluster_name']
            del data

            # Split pre-deafening PSDs into halves
            # 1st half will be used as basis and the 2nd half as control
            psd_list_basis, psd_list_pre, notes_basis, notes_pre = psd_split(psd_list_pre_all, notes_pre,
                                                                             context_list_pre)
            del psd_list_pre_all

            # Get basis psd and list of basis syllables
            db.cur.execute("SELECT songNote FROM main.cluster WHERE birdID = ?", (bird_id,))
            song_note = db.cur.fetchone()[0]

            psd_list_basis, note_list_basis = get_basis_psd(psd_list_basis, notes_basis, song_note=song_note,
                                                            num_note_crit_basis=num_note_crit)
            # Get similarity heatmap between basis and pre-deafening control
            get_similarity_heatmap(psd_list_pre, psd_list_basis, notes_pre, notes_basis, note_list_basis, file_list_pre,
                                   cluster_name_pre, save_results=save_heatmap)

        elif task_name == 'Postdeafening':

            if not 'psd_list_basis' in locals(): continue

            # data = select_context(data, context_selection=context_selection)

            # file list info is needed for this condition to track chronological changes
            psd_list_post, notes_post, file_list_post, context_list_post, cluster_name_post = \
                data['psd_list'], data['psd_notes'], data['file_list'], data['psd_context_list'], data['cluster_name']
            del data

            # Get similarity heatmap between basis and post-deafening
            similarity_info = \
                get_similarity_heatmap(psd_list_post, psd_list_basis, notes_post, notes_basis, note_list_basis,
                                       file_list_post, cluster_name_post, save_results=save_heatmap)

    # Get mean psd similarity per note
    query = f"SELECT * FROM cluster WHERE birdID = '{bird_id}' AND analysisOK = 1"
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        unit_nb = int(cluster_db.unit[-2:])
        channel_nb = int(cluster_db.channel[-2:])
        format = cluster_db.format
        song_note = cluster_db.songNote

        # Load class object
        ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object

        for i, note in enumerate(song_note):  # loop through notes

            # Get correlation between number of spikes and similarity
            ind = similarity_info[note]['note_cluster'] == ci.name
            if not ind.sum(): continue
            note_similarity = similarity_info[note]['similarity'][ind][:, i]
            mean_similarity = round(note_similarity.mean(), 3)

            if update_db:
                db.cur.execute(f"UPDATE syllable SET psdSimilarity = ({mean_similarity}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                db.conn.commit()

    if update_db:
        db.to_csv('syllable')
    print('Done!')



    # Get correlation between the number of spikes
    print('Get spk correlation')
    data_path = ProjectLoader().path / 'Analysis' / 'PSD_similarity' / 'SpkCount'  # the folder where spike info is stored

    # Load database
    # query = "SELECT * FROM cluster WHERE ephysOK = 1"

    query = f"SELECT * FROM cluster WHERE birdID = '{bird_id}' AND analysisOK = 1"
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        unit_nb = int(cluster_db.unit[-2:])
        channel_nb = int(cluster_db.channel[-2:])
        format = cluster_db.format
        song_note = cluster_db.songNote

        # Load class object
        ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object

        # Load number of spikes
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'PSD_similarity' + '/' + 'SpkCount',
                                  add_date=False)

        pre_motor_spk_dict = get_pre_motor_spk_per_note(ci, song_note, save_path, context_selection=context_selection,
                                                        npy_update=True)
        pre_motor_win = pre_motor_spk_dict['pre_motor_win']
        del pre_motor_spk_dict['pre_motor_win']

        # Plot the results
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(ci.name, y=.95)

        for i, note in enumerate(pre_motor_spk_dict):  # loop through notes

            # Get correlation between number of spikes and similarity
            ind = similarity_info[note]['note_cluster'] == ci.name
            if not ind.sum(): continue
            note_similarity = similarity_info[note]['similarity'][ind][:, i]
            mean_similarity = round(note_similarity.mean(), 3)
            spk_count = pre_motor_spk_dict[note]['nb_spk']
            pre_motor_fr = round(spk_count.sum() / (spk_count.shape[0] * (pre_motor_win / 1E3)),
                                 3)  # firing rates during the pre-motor window
            corr, corr_pval = pearsonr(note_similarity, spk_count)
            r_square = corr ** 2

            ax = plt.subplot2grid((nb_row, len(pre_motor_spk_dict.keys())), (1, i), rowspan=2, colspan=1)
            ax.scatter(spk_count, note_similarity, color='k', s=5)
            ax.set_title(note, size=font_size)
            if i == 0:
                ax.set_ylabel('Note similarity')
            ax.set_xlabel('Spk Count')
            # ax.set_ylim([0, 1])
            remove_right_top(ax)

            # Print out results
            ax_txt = plt.subplot2grid((nb_row, len(pre_motor_spk_dict.keys())), (3, i), rowspan=2, colspan=1)
            txt_xloc = 0
            txt_yloc = 0.5
            txt_inc = 0.2
            # ax_txt.set_ylim([0, 1])
            ax_txt.text(txt_xloc, txt_yloc, f"PremotorFR = {round(pre_motor_fr, 3)} (Hz)", fontsize=font_size)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 3)}", fontsize=font_size)
            txt_yloc -= txt_inc
            t = ax_txt.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 3)}", fontsize=font_size)
            if corr_pval < alpha:
                corr_sig = True
                t.set_bbox(dict(facecolor='green', alpha=0.5))
            else:
                corr_sig = False
                t.set_bbox(dict(facecolor='red', alpha=0.5))

            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"R_square = {round(r_square, 3)}", fontsize=font_size)
            ax_txt.axis('off')

            if update_db:
                with open('../../database/create_spk_corr.sql', 'r') as sql_file:
                    db.conn.executescript(sql_file.read())
                db.cur.execute("INSERT OR IGNORE INTO spk_corr (clusterID) VALUES (?)", (row["id"],))
                db.cur.execute(
                    "UPDATE spk_corr SET note=?, nbPremotorFR=?, corrR=?, corrPval=?, corrSig=?, rsquare=? WHERE clusterID=?",
                    (note, round(pre_motor_fr, 3), round(corr, 3), round(corr_pval, 3), corr_sig, round(r_square, 3),
                     row["id"]))
                db.conn.commit()

        # Save figure
        if save_fig and 'ax' in locals():
            save_path = save.make_dir(ProjectLoader().path / 'Analysis/PSD_similarity', 'SpkCorr')
            save.save_fig(fig, save_path, ci.name, fig_ext=fig_ext)
            del ax

    gc.collect()
    if update_db:
        db.to_csv('spk_corr')
    print('Done!')
db.conn.close()
