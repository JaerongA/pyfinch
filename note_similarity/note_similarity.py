from analysis.functions import *
from analysis.parameters import *
import scipy
from scipy import spatial
from scipy.io import wavfile
from matplotlib.pylab import psd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import pandas as pd
import seaborn as sns
from util.functions import *
from util.spect import *
from util.draw import *
from util import save
import json

# Parameters
font_size = 12  # figure font size
note_buffer = 10  # in ms before and after each note

num_note_crit_basis = 30  # the number of basis note should be >= this criteria
num_note_crit_testing = 10  # the number of testing syllables should be >= this criteria


# Obtain basis data from training files
def get_psd_mat(data_path, save_fig=False, nfft=2 ** 10):
    file_name = data_path / 'PSD.npy'

    # Read from a file if it already exists
    if file_name.exists():
        data = np.load(file_name, allow_pickle=True).item()
        psd_array, psd_list, all_notes = data['psd_array'], data['psd_list'], data['all_notes']
    else:

        # Load files
        files = list(data_path.glob('*.wav'))
        # files = files[:10]

        psd_list = []  # store psd vectors for training
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

                if save_fig:
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
                                                               vmin=0.5,
                                                               vmax=100
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
                    ax_psd.set_xticks([])
                    plt.setp(ax_psd.set_yticks([]))
                    # plt.show()

                    # Save figure
                    save_path = save.make_dir(file.parent, 'Spectrograms')
                    save.save_fig(fig, save_path, fig_name, ext='.png')

                all_notes += syllable
                psd_list.append(psd_power)

        psd_array = np.asarray(psd_list)  # number of syllables x psd

        # Organize data into a dictionary
        data = {
            'psd_array': psd_array,
            'psd_list': psd_list,
            'all_notes': all_notes,
        }
        # Save results
        np.save(file_name, data)

    return psd_array, psd_list, all_notes


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


# Store results in the dataframe
df = pd.DataFrame()

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
            psd_array_training, psd_list_training, notes_training = get_psd_mat(training_path, save_fig=False)

            # Get basis psds per note
            psd_basis_list, note_basis_list = get_basis_psd(psd_array_training, notes_training)

            # Get psd from the testing set
            psd_array_testing, psd_list_testing, notes_testing = get_psd_mat(testing_path, save_fig=False)

            # Get similarity per syllable
            # Get psd distance
            distance = scipy.spatial.distance.cdist(psd_list_testing, psd_basis_list,
                                                    'sqeuclidean')  # (number of notes x number of basis notes)

            # Convert to similarity matrices
            similarity = 1 - (distance / np.max(distance))  # (number of notes x number of basis notes)

            # Plot similarity matrix per syllable
            note_testing_list = unique(notes_testing)  # convert syllable string into a list of unique syllables

            # Remove non-syllables (e.g., '0')
            if '0' in note_testing_list:
                note_testing_list.remove('0')
            if condition == 'control' and 'x' in note_testing_list:  # remove 'x' if it appears in the control data
                note_testing_list.remove('x')

            for note in note_testing_list:

                fig = plt.figure(figsize=(5, 5))
                # title = "Sim matrix: note = {}".format(note)
                fig_name = f"note - {note}"
                gs = gridspec.GridSpec(7, 8)

                ax = plt.subplot(gs[0:5, 1:7])
                ind = find_str(notes_testing, note)
                note_similarity = similarity[ind, :]
                nb_note = len(ind)

                if nb_note < num_note_crit_testing:
                    continue

                title = f"Sim matrix: note = {note} ({nb_note})"
                ax = sns.heatmap(note_similarity,
                                 vmin=0,
                                 vmax=1,
                                 cmap='binary')
                ax.set_title(title)
                ax.set_ylabel('Test syllables')
                ax.set_xticklabels(note_basis_list)
                plt.tick_params(left=False)
                plt.yticks([0.5, nb_note - 0.5], ['1', str(nb_note)])

                # Get mean or meadian similarity index
                ax = plt.subplot(gs[-1, 1:7], sharex=ax)
                similarity_mean = np.expand_dims(np.mean(note_similarity, axis=0), axis=0)  # or axis=1
                similarity_median = np.expand_dims(np.median(note_similarity, axis=0), axis=0)  # or axis=1

                ax = sns.heatmap(similarity_mean, annot=True, cmap='binary', vmin=0, vmax=1, annot_kws={"fontsize": 7})
                ax.set_xlabel('Basis syllables')
                ax.set_yticks([])
                ax.set_xticklabels(note_basis_list)
                # plt.show()

                if note in note_basis_list:  # if the testing note is in the basis set
                    note_in_basis = True
                    similarity_mean_val = similarity_mean[0][note_basis_list.index(note)]
                    similarity_median_val = similarity_median[0][note_basis_list.index(note)]
                else:  # if it's a novel note, pick the max value
                    note_in_basis = False
                    similarity_mean_val = np.max(similarity_mean[0])
                    similarity_median_val = np.max(similarity_median[0])

                #TODO: Get entropy & softmax prob

                # Save figure
                save_path = save.make_dir(testing_path, 'NoteSimilarity', add_date=True)
                save.save_fig(fig, save_path, fig_name, ext='.png')

                # Save results to a dataframe
                temp_df = []
                temp_df = pd.DataFrame({'BirdID': bird,
                                        'Condition': condition,
                                        'Note': note,  # testing note
                                        'NoteInBasis': [note_in_basis],
                                        'NoteX': note is 'x',
                                        'NbNotes': [nb_note],
                                        'SimilarityMean': [similarity_mean_val],
                                        'SimilarityMedian': [similarity_median_val]
                                        })
                df = df.append(temp_df, ignore_index=True)

# Save to csv
df.index.name = 'Index'
outputfile = project_path / 'Results' / 'SimilarityIndex.csv'
df.to_csv(outputfile, index=True, header=True)  # save the dataframe to .cvs format
print('Done!')
