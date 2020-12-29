from matplotlib.pylab import psd
import matplotlib.pyplot as plt
from pathlib import Path
from analysis.functions import *
import sys as sys
import os as os
import scipy as sc
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal
from scipy import spatial
from matplotlib.pylab import psd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import sklearn as skl
from sklearn import cluster
from sklearn import metrics
from scipy import spatial
from sklearn.mixture import GaussianMixture as GMM
from sklearn import decomposition
import random as rnd
from array import array
import seaborn as sns
from util.functions import *
from util.spect import *
from util.draw import *
from util import save

from analysis.parameters import *

# Parameters
datano = -1  # use all syllables
font_size = 12  # figure font size
note_buffer = 10  # in ms before and after each note

# Data path
training_path = Path('H:\Box\Data\BMI\y3y18\pre-control1')
testing_path = Path('H:\Box\Data\BMI\y3y18\BMI')


def norm(a):
    """normalizes a string by it's average and sd"""
    a = (np.array(a) - np.average(a)) / np.std(a)
    return a


# Obtain basis data from training files
def get_psd_mat(path, fig_ok=False):
    # Load files
    psd_list = []  # store psd vectors for training
    all_syllables = ''

    files = list(path.glob('*.wav'))

    for file in files:

        notmat_file = file.with_suffix('.wav.not.mat')
        onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')
        sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
        length = data.shape[0] / sample_rate
        timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3, 3)  # start from t = 0 in ms, reduce floating precision
        list_zip = zip(onsets, offsets, syllables)

        for i, (onset, offset, syllable) in enumerate(list_zip):

            # Get spectrogram
            ind, _ = extract_ind(timestamp, [onset - note_buffer, offset + note_buffer])
            extracted_data = data[ind]
            spect, freqbins, timebins = spectrogram(extracted_data, sample_rate, freq_range=freq_range)

            # Get power spectral density
            # nfft = int(round(2 ** 14 / 32000.0 * sample_rate))
            nfft = 2 ** 10

            psd_seg = psd(norm(extracted_data), NFFT=nfft, Fs=sample_rate)  # PSD segment from the time range
            seg_start = int(round(freq_range[0] / (sample_rate / float(nfft))))  # 307
            seg_end = int(round(freq_range[1] / (sample_rate / float(nfft))))  # 8192
            psd_power = norm(psd_seg[0][seg_start:seg_end])
            psd_freq = psd_seg[1][seg_start:seg_end]

            if fig_ok:
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
                plt.close(fig)

            save_path = save.make_dir(file.parent, 'Spectrograms')
            save.save_fig(fig, save_path, fig_name, ext='.png')

            all_syllables += syllable
            psd_list.append(psd_power)

    psd_array = np.asarray(psd_list)  # power x number of syllables

    return psd_array, psd_list, all_syllables


psd_array_training, psd_list_training, all_syllables_training = get_psd_mat(training_path, fig_ok=True)





# Get avg psd from the training set (will serve as a basis)

psd_dict = {}
psd_list_basis = []
syl_list_basis = []

for syllable in unique(all_syllables_training):

    if syllable != 'x':
        ind = find_str(all_syllables_training, syllable)
        syl_pow_array = psd_array_training[ind, :]
        syl_pow_avg = syl_pow_array.mean(axis=0)
        # plt.plot(syl_pow_avg), plt.show()
        temp_dict = {syllable: syl_pow_avg}
        psd_list_basis.append(syl_pow_avg)
        syl_list_basis.append(syllable)
        psd_dict.update(temp_dict)  # basis
        # plt.plot(psd_dict[syllable])

# plt.show()

# basis_psd_arr = np.asarray(basis_psd_list)


## Get psd from the testing set
psd_array_testing, psd_list_testing, all_syllables_testing = get_psd_mat(testing_path)

## Get similarity per syllable

# for i, (psd, syllable) in enumerate(zip(psd_list_testing, all_syllables_testing)):
#
#     if i ==0:
#         syllable = all_syllables_testing[i]
#
#         for basis_psd, basis_syllable in psd_dict.items():
#
#             print("Basis {} vs. Test {}".format(basis_syllable, syllable))
#
#             distance = sc.spatial.distance.cdist(psd_list_testing, basis_psd_list, 'sqeuclidean')
#             print(distance)


distance = sc.spatial.distance.cdist(psd_list_testing, psd_list_basis,
                                     'sqeuclidean')  # (number of syllables x basis syllables)

# convert to similarity matrices:
similarity = 1 - (distance / np.max(distance))

# # plot similiarity matrix
# fig = plt.figure(figsize=(3,6))
# ax = plt.subplot(111)
#
# # ax =sns.heatmap(distance, cmap='hot_r')
# # ax =sns.heatmap(similarity[:100,:], vmin=0.2, vmax=1)
# ax = sns.heatmap(similarity[:100,:], cmap='hot_r')
# ax.set_title('Sim matrix')
# ax.set_ylabel('N sample PSDs')
# ax.set_xlabel('Basis syllables')
# ax.set_xticklabels(syl_list_basis)
# ax.set_yticks([])
# plt.show()


# # plot similiarity matrix (samples)
# fig = plt.figure(figsize=(3,6))
# ax = plt.subplot(111)
# ax = sns.heatmap(similarity[30:60,:], cmap='hot_r')
# ax.set_title('Sim matrix')
# ax.set_ylabel('Test syllables')
# ax.set_xlabel('Basis syllables')
# ax.set_xticklabels(syl_list_basis)
# ax.set_yticklabels(list(all_syllables_testing[30:60]))
# plt.yticks(rotation=0)
# plt.show()


# Plot similarity matrix per syllable


for syllable in unique(all_syllables_testing):
    if syllable != '0':
        print(syllable)

        fig = plt.figure(figsize=(5, 6))

        gs = gridspec.GridSpec(6, 1)
        ax = plt.subplot(gs[0:5])

        title = "Sim matrix: syllable = {}".format(syllable)
        ind = find_str(all_syllables_testing, syllable)
        similarity_syllable = similarity[ind, :]
        # ax = sns.heatmap(similarity_syllable, cmap='binary', vmin=0, vmax=1)
        ax = sns.heatmap(similarity_syllable, cmap='binary')
        # ax.imshow(similarity_syllable, cmap='hot_r')
        ax.set_title(title)
        ax.set_ylabel('Test syllables')
        ax.set_xlabel('Basis syllables')
        ax.set_xticklabels(syl_list_basis)
        plt.yticks(rotation=0)

        ax = plt.subplot(gs[-1], sharex=ax)
        similarity_vec = np.expand_dims(np.mean(similarity_syllable, axis=0), axis=0)  # or axis=1
        ax = sns.heatmap(similarity_vec, annot=True, cmap='binary', vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticklabels(syl_list_basis)

        plt.show()

        save_path = save.make_dir(testing_path, 'Spectrograms')
        save.save_fig(fig, save_path, title, ext='.png')
