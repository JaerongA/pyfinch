"""
By Jaerong
PSD similarity metrics obtained from David Mets
"""

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

from analysis.functions import *
from analysis.parameters import *
from util import save
from util.draw import *
from util.functions import *
from util.spect import *

# "birdID": ["g20r5", "y58y59", "k71o7", "y3y18", "o54w8", "k77r57", "b86g86"],

# "birdID": ["o54w8", "k77r57", "b86g86"]  # birds to make figures

# Parameters
font_size = 12  # figure font size
note_buffer = 10  # in ms before and after each note

num_note_crit_basis = 30  # the number of basis note should be >= this criteria
num_note_crit_testing = 10  # the number of testing syllables should be >= this criteria
fig_save_ok = False
file_save_ok = False
save_psd = True
update = True  # generate a new .npz file or update the file
fig_ext = '.png'  # figure file extension


def get_basis_psd(psd_array, notes):
    # Get avg psd from the training set (will serve as a basis)
    psd_dict = {}
    psd_basis_list = []
    syl_basis_list = []

    unique_note = unique(notes)  # convert note string into a list of unique syllables

    # Remove unidentifiable note (e.g., '0' or 'x')
    if '0' in unique_note:
        unique_note.remove('0')
    if 'x' in unique_note:
        unique_note.remove('x')

    for note in unique_note:
        ind = find_str(notes, note)
        if len(ind) >= num_note_crit_basis:  # number should exceed the  criteria
            syl_pow_array = psd_array[ind, :]
            syl_pow_avg = syl_pow_array.mean(axis=0)
            temp_dict = {note: syl_pow_avg}
            psd_basis_list.append(syl_pow_avg)
            syl_basis_list.append(note)
            psd_dict.update(temp_dict)  # basis
            # plt.plot(psd_dict[note])
            # plt.show()
    return psd_basis_list, syl_basis_list


# Obtain basis data from training files
def get_psd_mat(data_path, save_psd=False, update=False, open_folder=False, nfft=2 ** 10, fig_ext='.png'):
    # Read from a file if it already exists
    file_name = data_path / 'PSD.npy'

    if save_psd and not update:
        raise Exception("psd can only be save in an update mode or when the .npy does not exist!, set update to TRUE")

    if update or not file_name.exists():

        # Load files
        files = list(data_path.glob('*.wav'))
        # files = files[:10]

        psd_list = []  # store psd vectors for training
        file_list = []  # store files names containing psds
        all_notes = ''  # concatenate all syllables

        for file in files:

            notmat_file = file.with_suffix('.wav.not.mat')
            onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')
            sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
            length = data.shape[0] / sample_rate
            timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3,
                                 3)  # start from t = 0 in ms, reduce floating precision
            list_zip = zip(onsets, offsets, syllables)

            for i, (onset, offset, syllable) in enumerate(list_zip):

                # Get spectrogram
                ind, _ = extract_ind(timestamp, [onset - note_buffer, offset + note_buffer])
                extracted_data = data[ind]
                spect, freqbins, timebins = spectrogram(extracted_data, sample_rate, freq_range=freq_range)

                # Get power spectral density
                # nfft = int(round(2 ** 14 / 32000.0 * sample_rate))  # used by Dave Mets

                # Get psd after normalization
                psd_seg = psd(normalize(extracted_data), NFFT=nfft, Fs=sample_rate)  # PSD segment from the time range
                seg_start = int(round(freq_range[0] / (sample_rate / float(nfft))))  # 307
                seg_end = int(round(freq_range[1] / (sample_rate / float(nfft))))  # 8192
                psd_power = normalize(psd_seg[0][seg_start:seg_end])
                psd_freq = psd_seg[1][seg_start:seg_end]

                # Plt & save figure
                if save_psd:
                    # Plot spectrogram & PSD
                    fig = plt.figure(figsize=(3.5, 3))
                    fig_name = "{}, note#{} - {}".format(file.name, i, syllable)
                    fig.suptitle(fig_name, y=0.95)
                    gs = gridspec.GridSpec(6, 3)

                    # Plot spectrogram
                    ax_spect = plt.subplot(gs[1:5, 0:2])
                    ax_spect.pcolormesh(timebins * 1E3, freqbins, spect,  # data
                                        cmap='hot_r',
                                        norm=colors.SymLogNorm(linthresh=0.05,
                                                               linscale=0.03,
                                                               vmin=0.5, vmax=100
                                                               ))

                    remove_right_top(ax_spect)
                    ax_spect.set_ylim(freq_range[0], freq_range[1])
                    ax_spect.set_xlabel('Time (ms)', fontsize=font_size)
                    ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)

                    # Plot psd
                    ax_psd = plt.subplot(gs[1:5, 2], sharey=ax_spect)
                    ax_psd.plot(psd_power, psd_freq, 'k')
                    ax_psd.spines['right'].set_visible(False), ax_psd.spines['top'].set_visible(False)
                    ax_psd.spines['bottom'].set_visible(False)
                    ax_psd.set_xticks([])  # remove xticks
                    plt.setp(ax_psd.set_yticks([]))
                    # plt.show()

                    # Save figures
                    save_path = save.make_dir(file.parent, 'Spectrograms')
                    save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=open_folder)
                    plt.close(fig)

                all_notes += syllable
                psd_list.append(psd_power)
                file_list.append(file.name)

        psd_array = np.asarray(psd_list)  # number of syllables x psd

        # Organize data into a dictionary
        data = {
            'psd_array': psd_array,
            'psd_list': psd_list,
            'file_list': file_list,
            'all_notes': all_notes,
        }
        # Save results
        np.save(file_name, data)

    else:  # if not update or file already exists
        data = np.load(file_name, allow_pickle=True).item()
        psd_array, psd_list, file_list, all_notes = data['psd_array'], data['psd_list'], data['file_list'], data[
            'all_notes']

    return psd_array, psd_list, file_list, all_notes


# Store results in the dataframe
df = pd.DataFrame()
df_x = pd.DataFrame()
df_sig_prob = pd.DataFrame()  # dataframe for significant syllables

# Data path (Read from .json config file)
config_file = 'config.json'
with open(config_file, 'r') as f:
    config = json.load(f)

project_path = Path(config['project_dir'])

for bird in config['birdID']:

    training_path = ''

    for session in config['sessions']:

        testing_path = ''
        condition = ''

        data_path = project_path / bird / session

        if session == "pre-control1":
            training_path = data_path
            # print(f"training path = {training_path}")
        else:
            testing_path = data_path
            # print(f"testing path = {testing_path}")

        if training_path and testing_path:
            if training_path.name == "pre-control1" and testing_path.name == "pre-control2":
                condition = 'baseline'
            elif training_path.name == "pre-control1" and testing_path.name == "BMI":
                condition = 'BMI'

        if condition:
            print(f"training path = {training_path}")
            print(f"testing path = {testing_path}")
            print(condition)
            print("")

            # Obtain basis data from training files
            psd_array_training, psd_list_training, file_list_training, notes_training = get_psd_mat(training_path,
                                                                                                    update=update,
                                                                                                    save_psd=save_psd,
                                                                                                    fig_ext=fig_ext)

            # Get basis psds per note
            psd_basis_list, note_basis_list = get_basis_psd(psd_array_training, notes_training)

            # Get psd from the testing set
            psd_array_testing, psd_list_testing, file_list_testing, notes_testing = get_psd_mat(testing_path,
                                                                                                update=update,
                                                                                                save_psd=save_psd,
                                                                                                fig_ext=fig_ext)

            # Get similarity per syllable
            # Get psd distance
            distance = scipy.spatial.distance.cdist(psd_list_testing, psd_basis_list,
                                                    'sqeuclidean')  # (number of test notes x number of basis notes)

            # Convert to similarity matrices
            similarity = 1 - (distance / np.max(distance))  # (number of test notes x number of basis notes)

            # Plot similarity matrix per syllable
            note_testing_list = unique(notes_testing)  # convert syllable string into a list of unique syllables

            # Remove non-syllables (e.g., '0')
            if '0' in note_testing_list:
                note_testing_list.remove('0')
            if condition == 'control' and 'x' in note_testing_list:  # remove 'x' if it appears in the control data
                note_testing_list.remove('x')

            # Get similarity matrix per test note
            for note in note_testing_list:

                if note not in note_basis_list and note != 'x':
                    continue

                ind = find_str(notes_testing, note)
                nb_note = len(ind)
                if nb_note < num_note_crit_testing:
                    continue

                # Get similarity matrix per note
                note_similarity = similarity[ind, :]  # number of the test notes x basis note

                # Get mean or median similarity index
                similarity_mean = np.expand_dims(np.mean(note_similarity, axis=0), axis=0)  # or axis=1
                similarity_sem = sem(note_similarity, ddof=1)
                similarity_median = np.expand_dims(np.median(note_similarity, axis=0), axis=0)  # or axis=1

                # Get entropy per note (per row)
                # Convert to probability distribution first
                # note_similarity_prob = (note_similarity / note_similarity.sum(axis=1, keepdims=True))
                # note_similarity_entropy = (-note_similarity_prob * np.log2(note_similarity_prob)).sum(axis=1)
                # note_similarity_entropy = round(np.nanmean(note_similarity_entropy), 3)

                # Get entropy from the mean similarity index
                note_similarity_prob = (similarity_mean / similarity_mean.sum(axis=1, keepdims=True))
                note_similarity_entropy = (-note_similarity_prob * np.log2(note_similarity_prob)).sum(axis=1)
                note_similarity_entropy = round(np.nanmean(note_similarity_entropy), 3)

                # Plot the similarity matrix
                fig = plt.figure(figsize=(5, 5))
                # title = "Sim matrix: note = {}".format(note)
                fig_name = f"note - {note}"
                title = f"Sim matrix: note = {note} ({nb_note}), entropy = {note_similarity_entropy}"
                gs = gridspec.GridSpec(7, 8)
                ax = plt.subplot(gs[0:5, 1:7])
                ax = sns.heatmap(note_similarity,
                                 vmin=0, vmax=1,
                                 cmap='binary')
                ax.set_title(title)
                ax.set_ylabel('Test syllables')
                ax.set_xticklabels(note_basis_list)
                plt.tick_params(left=False)
                plt.yticks([0.5, nb_note - 0.5], ['1', str(nb_note)])

                ax = plt.subplot(gs[-1, 1:7], sharex=ax)

                ax = sns.heatmap(similarity_mean, annot=True, cmap='binary',
                                 vmin=0, vmax=1,
                                 annot_kws={"fontsize": 7})
                ax.set_xlabel('Basis syllables')
                ax.set_yticks([])
                ax.set_xticklabels(note_basis_list)
                # plt.show()

                if note is 'x':  # get the max if 'x'
                    similarity_mean_val = np.max(similarity_mean[0])
                    similarity_median_val = np.max(similarity_median[0])
                else:  # get the value from the matching note
                    similarity_mean_val = similarity_mean[0][note_basis_list.index(note)]
                    similarity_median_val = similarity_median[0][note_basis_list.index(note)]

                # Save figure
                if fig_save_ok:
                    save_path = save.make_dir(testing_path, 'NoteSimilarity', add_date=True)
                    save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=False)
                else:
                    plt.close(fig)

                # Save results to a dataframe
                # All notes
                temp_df = []
                temp_df = pd.DataFrame({'BirdID': bird,
                                        'Condition': condition,
                                        'Note': note,  # testing note
                                        'NoteX': note is 'x',
                                        'NbNotes': [nb_note],
                                        'SimilarityMean': [similarity_mean_val],
                                        'SimilarityMedian': [similarity_median_val],
                                        'Entropy': [note_similarity_entropy]
                                        })
                df = df.append(temp_df, ignore_index=True)

                # 'x' in BMI condition only
                if condition == 'BMI' and note is 'x':  # store mean similarity values for 'x'
                    for ind, basis_note in enumerate(note_basis_list):
                        temp_df_x = []
                        temp_df_x = pd.DataFrame({'BirdID': bird,
                                                  'BasisNote': basis_note,  # testing note
                                                  'SimilarityMean': [similarity_mean[0][ind]],
                                                  'SimilaritySEM': [similarity_sem[ind]],
                                                  })
                        df_x = df_x.append(temp_df_x, ignore_index=True)

    # Calculate the proportion of 'x's that exceeds the mean value of the similarity matrix in the control condition
    # 'x' similarity matrix
    ind = find_str(notes_testing, 'x')
    nb_note = len(ind)
    if len(ind) < num_note_crit_testing:  # if there are not enough 'x's, skip
        continue

    new_df = df.groupby(['BirdID', 'Condition'])['SimilarityMean'].mean().reset_index()
    new_df = new_df[(new_df['Condition'] == 'baseline') & (new_df['BirdID'] == bird)]

    x_similarity = similarity[ind, :]  # number of the test notes x basis note
    sim_basis_mean = new_df['SimilarityMean'].values
    # proportion of notes having a higher similarity index relative to the baseline (mean similarity ind from the control (pre1 vs. pre2)
    # the total number of cells as a denominator
    # prob_sig_notes = (x_similarity > sim_basis_mean[0]).sum() / x_similarity.size

    # the number of x's having at least one significant note as a denominator
    non_zero_prob = np.nonzero((x_similarity > sim_basis_mean[0]).sum(axis=1))[0].shape
    prob_sig_notes = non_zero_prob[0] / x_similarity.shape[0]

    # Select the maximum note only
    # max_col = x_similarity[:, x_similarity.mean(axis=0).argmax()]
    # prob_sig_notes = (max_col > sim_basis_mean[0]).sum() / max_col.size  # proportion of notes having a higher similarity index relative to the baseline (mean similarity ind from the control (pre1 vs. pre2)

    # Save results to a dataframe
    # All notes
    temp_df = []
    temp_df = pd.DataFrame({'BirdID': bird,
                            'SigProportion': [prob_sig_notes],
                            })
    df_sig_prob = df_sig_prob.append(temp_df, ignore_index=True)

    # Plot x
    frame_width = 2
    fig = plt.figure(figsize=(8, 5))
    title = f"{bird} SimilarityMat (x) - SigProb = {round(prob_sig_notes, 3)}, baseline SI = {round(sim_basis_mean[0], 3)}"
    plt.suptitle(title, size=15)
    fig_name = f"{bird}_SimilarityMat(x)"
    gs = gridspec.GridSpec(8, 10)
    ax = plt.subplot(gs[1:7, 1:5])
    ax = sns.heatmap(x_similarity,
                     vmin=0, vmax=1,
                     cmap='binary')

    ax.set_ylabel('Test syllables')
    ax.set_xticklabels(note_basis_list)
    plt.tick_params(left=False)
    plt.yticks([0.5, nb_note - 0.5], ['1', str(nb_note)])

    # Plot with a boolean mask (sig bins only)
    x_similarity[x_similarity < sim_basis_mean[0]] = np.nan  # replace non-sig values with nan

    ax = plt.subplot(gs[1:7, 5:-1])
    ax = sns.heatmap(x_similarity,
                     vmin=0, vmax=1,
                     cmap='binary')

    ax.axhline(y=0, color='k', linewidth=frame_width / 4)
    ax.axhline(y=x_similarity.shape[0], color='k', linewidth=frame_width)
    ax.axvline(x=0, color='k', linewidth=frame_width / 4)
    ax.axvline(x=x_similarity.shape[1], color='k', linewidth=frame_width)

    ax.set_xticklabels(note_basis_list)
    ax.set_yticklabels([])
    plt.tick_params(left=False)
    # plt.show()

    # Save the figure
    save_path = save.make_dir(project_path / 'Results', 'SigProb', add_date=True)
    save.save_fig(fig, save_path, fig_name, fig_ext='.png')

    # Save the x similarity matrix (with nans) in .csv for visual verification (01/17/2021)
    x_csv_path = project_path / 'Results' / f'{bird}_SimilarityMat(x).csv'
    temp_df = []
    temp_df = pd.DataFrame(np.round(x_similarity, 3), columns=note_basis_list)
    file_array = np.asarray(file_list_testing)
    temp_df['File'] = file_array[ind].tolist()
    temp_df.to_csv(x_csv_path, index=True, header=True)  # save the dataframe to .cvs format

# Save to csv (sig proportion)
outputfile = project_path / 'Results' / 'SigProportion.csv'
df_sig_prob.to_csv(outputfile, index=True, header=True)  # save the dataframe to .cvs format

# Save to csv
if file_save_ok:
    df.index.name = 'Index'
    outputfile = project_path / 'Results' / 'SimilarityIndex.csv'
    df.to_csv(outputfile, index=True, header=True)  # save the dataframe to .cvs format

    df_x.index.name = 'Index'
    outputfile = project_path / 'Results' / 'NoteX.csv'
    df_x.to_csv(outputfile, index=True, header=True)  # save the dataframe to .cvs format

print('Done!')
