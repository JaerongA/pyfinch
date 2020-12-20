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
    out.append(sc.signal.filtfilt(b[0], b[1], a[0]))
    out.append(a[1])
    return (out)


def threshold(a, thresh=None):
    """Returns a thresholded array of the same length as input
 with everything below a specific threshold set to 0.
 By default threshold is sigma."""
    if thresh == None: thresh = sc.std(a)
    out = np.where(abs(a) > thresh, a, np.zeros(a.shape))
    return out


def findobject(file):
    """finds objects.  Expects a smoothed rectified amplitude envelope"""
    value = (otsu(np.array(file, dtype=np.uint32))) / 2  # calculate a threshold
    # value=(np.average(file))/2 #heuristically, this also usually works  for establishing threshold
    thresh = threshold(file, value)  # threshold the envelope data
    thresh = threshold(sc.ndimage.convolve(thresh, np.ones(512)), 0.5)  # pad the threshold
    label = (sc.ndimage.label(thresh)[0])  # label objects in the threshold
    objs = sc.ndimage.find_objects(label)  # recover object positions
    return (objs)


def smoothrect(a, window=None, freq=None):
    """smooths and rectifies a song.  Expects (data,samprate)"""
    if freq == None: freq = 32000  # baseline values if none are provided
    if window == None: window = 2  # baseline if none are provided
    le = int(round(freq * window / 1000))  # calculate boxcar kernel length
    h = np.ones(le) / le  # make boxcar
    smooth = np.convolve(h, abs(a))  # convovlve boxcar with signal
    offset = int(round((len(smooth) - len(a)) / 2))  # calculate offset imposed by convolution
    smooth = smooth[(1 + offset):(len(a) + offset)]  # correct for offset
    return smooth


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

training_path = Path('H:\Box\Data\BMI\k71o7\TrainingSet(withX)')
testing_path = Path('H:\Box\Data\BMI\k71o7\TestSet2')

training_files = list(training_path.glob('*.wav'))
testing_Files = list(testing_path.glob('*.wav'))
#
# training_files = [x for x in os.listdir(path1) if x[-4:] == '.wav']
# testing_Files = [x for x in os.listdir(path2) if x[-4:] == '.wav']

# Load files

for file in training_files:

    notmat_file = file.with_suffix('.wav.not.mat')

    onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')

    sample_rate, data = wavfile.read(file)  # note that the timestamp is in second

    length = data.shape[0] / sample_rate
    timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3,
                         3)  # start from t = 0 in ms, reduce floating precision

    list_zip = zip(onsets, offsets, syllables)

    for onset, offset, syllable in list_zip:

        # data_list =
        # ind = extract_ind(timestamp, [onset - note_buffer, offset + note_buffer])
        ind = extract_ind(timestamp, [onset, offset])
        extracted_data = data[ind]
        spect, freqbins, timebins = spectrogram(extracted_data, sample_rate, freq_range=freq_range)

        # Plot spectrogram
        ax_spect = plt.subplot(111)
        ax_spect.pcolormesh(timebins * 1E3, freqbins, spect,  # data
                            cmap='hot_r',
                            norm=colors.SymLogNorm(linthresh=0.05,
                                                   linscale=0.03,
                                                   vmin=0.5,
                                                   vmax=100
                                                   ))

        remove_right_top(ax_spect)
        ax_spect.set_ylim(freq_range[0], freq_range[1])
        ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)

        break