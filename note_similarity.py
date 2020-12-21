from matplotlib.pylab import psd
import matplotlib.pyplot as plt
from pathlib import Path
from song.analysis import *
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
from  util.functions import *
from  util.spect import *
from util.draw import *
from util import save

from song.parameters import *


# functions

def impwav(a):
    """Imports a wave file as an array where a[1]
 is the sampling frequency and a[0] is the data"""
    out = []
    wav = sc.io.wavfile.read(a)
    out = [wav[1], wav[0]]
    return out


def norm(a):
    """normalizes a string by it's average and sd"""
    a = (np.array(a) - np.average(a)) / np.std(a)
    return a


def filtersong(a):
    """highpass iir filter for song."""
    out = []
    b = sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60, ftype='ellip')
    out.append(sc.signal.filtfilt(b[0], b[1], a))
    # out.append(a[1])
    return (out)



def getsyls(a):
    """takes a file read in with impwav and returns a list of sylables"""
    fa = filtersong(a)  # filter song input
    frq = a[1]  # get sampling frequency from data (a)
    a = a[0]  # get data from data (a)
    frqs = frq / 1000  # calcualte length of a ms in samples
    objs = findobject(smoothrect(fa[0], 10, frq))  # get syllable positions
    sylables = [x for x in [a[y] for y in objs] if
                int(len(x)) > (10 * frqs)]  # get syllable data if of sufficient duration
    '''uncomment the next line to recover syllables that have been high pass filtered as opposed to raw data.
 Using data filtered prior to PSD calculation helps if you data are contaminated
 with low frequency noise'''
    # sylables=[x for x in [fa[0][y] for y in objs] if int(len(x))>(10*frqs)] #get syllable data if of sufficient duration.
    objs = [y for y in objs if int(len(a[y])) > 10 * frqs]  # get objects of sufficient duration
    return sylables, objs, frq


datano = -1  # use all syllables
font_size = 12

# convert syllables into PSDs
# segedpsds = []  # psd of segmented syllables
# for x in syls[:datano]:
#     fs = x[0]
#     nfft = int(round(2 ** 14 / 32000.0 * fs))
#     segstart = int(round(600 / (fs / float(nfft))))  # 307
#     segend = int(round(16000 / (fs / float(nfft))))  # 8192
#     psds = [psd(norm(y), NFFT=nfft, Fs=fs) for y in x[1:]]  # get psd from song segment
#     spsds = [norm(n[0][segstart:segend]) for n in psds]
#     for n in spsds: segedpsds.append(n)
#     if len(segedpsds) > datano: break

note_buffer = 10  # in ms before and after each note
fig_ok = False

training_path = Path('H:\Box\Data\BMI\k71o7\TrainingSet(withX)')
testing_path = Path('H:\Box\Data\BMI\k71o7\TestSet2')



## Obtain basis data from training files

def get_psd_mat(path, fig_ok=False):

    # Load files
    psd_list = []  # store psd vectors for training
    all_syllables = ''

    files = list(path.glob('*.wav'))

    for file in files:

        notmat_file = file.with_suffix('.wav.not.mat')

        onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')

        sample_rate, data = wavfile.read(file)  # note that the timestamp is in second

        # data = filtersong(data)
        # filt = sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60, ftype='ellip')
        # data = sc.signal.filtfilt(filt[0], filt[1], data)


        length = data.shape[0] / sample_rate
        timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3,
                             3)  # start from t = 0 in ms, reduce floating precision

        list_zip = zip(onsets, offsets, syllables)

        for i, (onset, offset, syllable) in enumerate(list_zip):

            ind = extract_ind(timestamp, [onset - note_buffer, offset + note_buffer])
            # ind = extract_ind(timestamp, [onset, offset])
            extracted_data = data[ind]

            spect, freqbins, timebins = spectrogram(extracted_data, sample_rate, freq_range=freq_range)


            # Get power spectral density

            nfft = int(round(2 ** 14 / 32000.0 * sample_rate))
            # nfft = 2**10
            psd_seg = psd(norm(extracted_data), NFFT=nfft, Fs=sample_rate)

            segstart = int(round(freq_range[0] / (sample_rate / float(nfft))))  # 307
            segend = int(round(freq_range[1] / (sample_rate / float(nfft))))  # 8192
            pow = norm(psd_seg[0][segstart:segend])
            freq = psd_seg[1][segstart:segend]

            if fig_ok:

                # Plot spectrogram & PSD
                fig = plt.figure(figsize=(3.5, 3))
                fig_name = "{}, note#{} - {}".format(file.name, i, syllable)
                fig.suptitle(fig_name, y=0.95)

                gs = gridspec.GridSpec(6, 3)

                ax_spect = plt.subplot(gs[1:5,0:2])
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

                ax_psd = plt.subplot(gs[1:5,2], sharey=ax_spect)

                # pow = np.mean(spect, axis=1)  # time-resolved average
                freq = freqbins
                ax_psd.plot(pow, freq, 'k')

                ax_psd.spines['right'].set_visible(False), ax_psd.spines['top'].set_visible(False)
                ax_psd.spines['bottom'].set_visible(False)
                ax_psd.set_xticks([])
                plt.setp(ax_psd.set_yticks([]))
                plt.close(fig)

            # save_path = save.make_dir(file.parent, 'Spectrograms')
            # save.save_fig(fig, save_path, fig_name, ext='.png')

            all_syllables += syllable
            # psd_array = np.hstack((psd_array, pow))
            psd_list.append(pow)
        #
        #     break

    psd_array = np.asarray(psd_list)  # power x number of syllables

    return psd_array, psd_list, all_syllables



psd_array_training, psd_list_training, all_syllables_training = get_psd_mat(training_path)

## Get avg psd from the training set (will serve as a basis)

psd_dict = {}
psd_list_basis = []
syl_list_basis = []

for syllable in unique(all_syllables_training):

    if syllable != 'x':
        ind  = find_str(all_syllables_training, syllable)
        syl_pow_array = psd_array_training[ind,:]
        syl_pow_avg= syl_pow_array.mean(axis=0)
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


distance = sc.spatial.distance.cdist(psd_list_testing, psd_list_basis, 'sqeuclidean')  # (number of syllables x basis syllables)

#convert to similarity matrices:
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