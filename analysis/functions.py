"""
By Jaerong
A collection of functions used for song analysis
"""

from analysis.parameters import *
from matplotlib.pylab import psd
from util import save
from util.draw import *
from util.functions import *
from util.spect import *

def read_not_mat(notmat, unit='ms'):
    """ read from .not.mat files generated from uisonganal
    Parameters
    ----------
    notmat : path
        Name of the .not.mat file (path)
    unit : (optional)
        milli-second by default. Convert to seconds when specified

    Returns
    -------
    onsets : array
        time stamp for syllable onset (in ms)
    offsets : array
        time stamp for syllable offset (in ms)
    intervals : array
        temporal interval between syllables (i.e. syllable gaps) (in ms)
    durations : array
        durations of each syllable (in ms)
    syllables : str
        song syllables
    contexts : str
        social context ('U' for undirected and 'D' for directed)
    """
    import scipy.io
    onsets = scipy.io.loadmat(notmat)['onsets'].transpose()[0]  # syllable onset timestamp
    offsets = scipy.io.loadmat(notmat)['offsets'].transpose()[0]  # syllable offset timestamp
    intervals = onsets[1:] - offsets[:-1]  # syllable gap durations (interval)
    durations = offsets - onsets  # duration of each syllable
    syllables = scipy.io.loadmat(notmat)['syllables'][0]  # Load the syllable info
    contexts = notmat.name.split('.')[0].split('_')[-1][
        0].upper()  # extract 'U' (undirected) or 'D' (directed) from the file name
    if contexts not in ['U', 'D']:  # if the file was not tagged with Undir or Dir
        contexts = None

    # units are in ms by default, but convert to second with the argument
    if unit is 'second':
        onsets /= 1E3
        offsets /= 1E3
        intervals /= 1E3
        durations /= 1E3

    return onsets, offsets, intervals, durations, syllables, contexts


def syl_type_(syllables, song_info):
    """ function to determine the category of the syllable
    INPUT1: syllables (str)
    INPUT2: intervals (array) syllable gap duration
    OUTPUT: type of the syllable (e.g., motif, intro notes, calls)
    """
    syl_type = []
    for syllable in syllables:
        if syllable in song_info['motif']:
            syl_type.append('M')  # motif
        elif syllable in song_info['calls']:
            syl_type.append('C')  # call
        elif syllable in song_info['introNotes']:
            syl_type.append('I')  # intro notes
        else:
            syl_type.append(None)
    return syl_type


def demarcate_bout(target, intervals):
    """ Demarcate the song bout with an asterisk (*) from a string of syllables
    Parameters
    ----------
    target : str or numpy array
    intervals_ms : int
        syllable gap duration in ms

    Returns
    -------
    bout_labeling : str
        demarcated syllable string (e.g., 'iiiabc*abckn*')
    """
    from analysis.parameters import bout_crit
    import numpy as np

    ind = np.where(intervals > bout_crit)[0]
    bout_labeling = target

    if isinstance(target, str):
        if len(ind):
            for i, item in enumerate(ind):
                if i is 0:
                    bout_labeling = target[:item + 1]
                else:
                    bout_labeling += '*' + target[ind[i - 1] + 1:ind[i] + 1]
            bout_labeling += '*' + target[ind[i] + 1:]

        bout_labeling += '*'  # end with an asterisk


    elif isinstance(target, np.ndarray):
        if len(ind):
            for i, item in enumerate(ind):
                if i is 0:
                    bout_labeling = target[:item + 1]
                else:
                    bout_labeling = np.append(bout_labeling, '*')
                    bout_labeling = np.append(bout_labeling, target[ind[i - 1] + 1: ind[i] + 1])

            bout_labeling = np.append(bout_labeling, '*')
            bout_labeling = np.append(bout_labeling, target[ind[i] + 1:])

        bout_labeling = np.append(bout_labeling, '*')  # end with an asterisk

    return bout_labeling


def unique_nb_notes_in_bout(note: str, bout: str):
    """ returns the unique number of notes within a single bout string """
    nb_song_note_in_bout = len([note for note in note if note in bout])
    return nb_song_note_in_bout


def total_nb_notes_in_bout(note: str, bout: str):
    """ returns the total number of song notes from a list of song bouts"""
    notes = []
    nb_notes = []
    for note in note:
        notes.append(note)
        nb_notes.append(sum([bout.count(note) for bout in bout]))
    return sum(nb_notes)


def get_nb_bouts(song_note: str, bout_labeling: str):
    """ Count the number of bouts (only includes those having at least one song note)
    INPUT1: song_note (e.g., abcd, syllables that are part of a motif)
    INPUT2: bout_labeling (e.g., iiiiiiiiabcdjiiiabcdji*, syllables that are demarcated by * (bout))
    """
    nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if
                    unique_nb_notes_in_bout(song_note, bout)])
    return nb_bouts


def get_dur():
    """
    Get note & interval duration (mean or median)
    Returns:

    """
    pass


def get_snr(avg_wf, raw_neural_trace):
    """
    Calculate signal-to-noise ratio of the spike
    Parameters
    ----------
    avg_wf : array
        averaged spike waveform of a neuron
    raw_neural_trace : array
        raw neural signal
    Returns
    -------
    snr : float
        signal-to-noise ratio
    """

    import numpy as np

    snr = 10 * np.log10(np.var(avg_wf) / np.var(raw_neural_trace))  # in dB
    snr = round(snr, 3)
    return snr

def get_half_width(wf_ts, avg_wf):

    import numpy as np

    # Find the negative (or positive if inverted) deflection
    if np.argmin(avg_wf) > np.argmax(avg_wf):  # inverted waveform (peak comes first in extra-cellular recording)
        deflection_baseline = np.abs(avg_wf).mean() + np.abs(avg_wf).std(axis=0)  # the waveform baseline. Finds values above the baseline
    else:
        deflection_baseline = avg_wf.mean() - avg_wf.std(axis=0)  # the waveform baseline. Finds values below the baseline

    diff_ind = []
    for ind in np.where(np.diff(avg_wf > deflection_baseline))[0]:  # below mean amp
        diff_ind.append(ind)

    # Get the deflection range where the peak is detected
    max_amp_arr = []
    for ind, _ in enumerate(diff_ind):
        if ind < len(diff_ind) - 1:
            max_amp = np.max(np.abs(avg_wf[diff_ind[ind]: diff_ind[ind + 1]]))
            max_amp_arr = np.append(max_amp_arr, max_amp)

    # Get the absolute amp value of the peak and trough
    peak_trough_amp = np.sort(max_amp_arr)[::-1][0:2]

    # Decide which one is the peak
    peak_trough_ind = np.array([], dtype=np.int)

    for amp in peak_trough_amp:
        peak_trough_ind = np.append(peak_trough_ind, np.where(np.abs(avg_wf) == amp)[0])

    peak_ind = peak_trough_ind.min()  # note that the waveforms are inverted in extracelluar recording so peak comes first
    trough_ind = peak_trough_ind.max()

    deflection_range = []

    for ind, _ in enumerate(diff_ind):
        if ind < len(diff_ind) - 1:
            range = diff_ind[ind: ind + 2]
            if range[0] < peak_ind < range[1]:
                deflection_range = range
                break

    half_width = (wf_ts[deflection_range[1]] - wf_ts[deflection_range[0]]) / 2
    half_width *= 1E3  # convert to microsecond
    return deflection_range, round(half_width, 3)


def get_psd_mat(data_path, save_path,
                save_psd=False, update=False, open_folder=False, add_date=False,
                nfft=2 ** 10, fig_ext='.png'):

    import numpy as np
    from scipy.io import wavfile
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec

    # Parameters
    note_buffer = 20  # in ms before and after each note
    font_size = 12  # figure font size

    # Read from a file if it already exists
    file_name = data_path / 'PSD.npy'

    if save_psd and not update:
        raise Exception("psd can only be save in an update mode or when the .npy does not exist!, set update to TRUE")

    if update or not file_name.exists():

        # Load files
        files = list(data_path.glob('*.wav'))

        psd_list = []  # store psd vectors for training
        file_list = []  # store files names containing psds
        psd_notes = ''  # concatenate all syllables

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
                    fig.suptitle(fig_name, y=0.95, fontsize=10)
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
                    # ax_psd.spines['bottom'].set_visible(False)
                    # ax_psd.set_xticks([])  # remove xticks
                    plt.setp(ax_psd.set_yticks([]))
                    # plt.show()

                    # Save figures
                    save_path = save.make_dir(save_path, add_date=add_date)
                    save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, open_folder=open_folder)
                    plt.close(fig)

                psd_notes += syllable
                psd_list.append(psd_power)
                file_list.append(file.name)

        psd_array = np.asarray(psd_list)  # number of syllables x psd

        # Organize data into a dictionary
        data = {
            'psd_array': psd_array,
            'psd_list': psd_list,
            'file_list': file_list,
            'psd_notes': [psd_notes],
        }
        # Save results
        np.save(file_name, data)

    else:  # if not update or file already exists
        data = np.load(file_name, allow_pickle=True).item()
        psd_array, psd_list, file_list, psd_notes = data['psd_array'], data['psd_list'], data['file_list'], data[
            'psd_notes']

    return psd_array, psd_list, file_list, psd_notes
