"""
By Jaerong
A collection of functions used for song analysis
"""

# from analysis.parameters import *
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

    from analysis.parameters import freq_range
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
        psd_context_list = []  # concatenate syllable contexts

        for file in files:

            notmat_file = file.with_suffix('.wav.not.mat')
            onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')
            sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
            length = data.shape[0] / sample_rate
            timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3, 3)  # start from t = 0 in ms, reduce floating precision
            contexts = contexts * len(syllables)
            list_zip = zip(onsets, offsets, syllables, contexts)

            for i, (onset, offset, syllable, context) in enumerate(list_zip):

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
                    fig_name = "{}, note#{} - {} - {}".format(file.name, i, syllable, context)
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

                psd_list.append(psd_power)
                file_list.append(file.name)
                psd_notes += syllable
                psd_context_list.append(context)

        # Organize data into a dictionary
        data = {
            'psd_list': psd_list,
            'file_list': file_list,
            'psd_notes': psd_notes,
            'psd_context': psd_context_list,
        }
        # Save results
        np.save(file_name, data)

    else:  # if not update or file already exists
        data = np.load(file_name, allow_pickle=True).item()

    return data['psd_list'], data['file_list'], data['psd_notes'], data['psd_context']


def get_basis_psd(psd_list, notes, song_note=None, num_note_crit_basis=30):
    """
    Get avg psd from the training set (will serve as a basis)
    Parameters
    ----------
    psd_list : list
        List of syllable psds
    notes : str
        String of all syllables
    song_note : str
        String of all syllables
    num_note_crit_basis : int (30 by default)
        Minimum number of notes required to be a basis syllable

    Returns
    -------
    psd_list_basis : list
    note_list_basis : list
    """

    psd_dict = {}
    psd_list_basis = []
    note_list_basis = []

    psd_array = np.asarray(psd_list)   # number of syllables x psd (get_basis_psd function accepts array format only)
    unique_note = unique(''.join(sorted(notes)))  # convert note string into a list of unique syllables

    # Remove unidentifiable note (e.g., '0' or 'x')
    if '0' in unique_note:
        unique_note.remove('0')
    if 'x' in unique_note:
        unique_note.remove('x')

    for note in unique_note:
        if note not in song_note: continue
        ind = find_str(notes, note)
        if len(ind) >= num_note_crit_basis:  # number should exceed the  criteria
            note_pow_array = psd_array[ind, :]
            note_pow_avg = note_pow_array.mean(axis=0)
            temp_dict = {note: note_pow_avg}
            psd_list_basis.append(note_pow_avg)
            note_list_basis.append(note)
            psd_dict.update(temp_dict)  # basis
            # plt.plot(psd_dict[note])
            # plt.show()
    return psd_list_basis, note_list_basis


def get_pre_motor_spk_per_note(ClusterInfo, song_note, save_path,
                               context_selection=None,
                               npy_update=False):
    """
    Get the number of spikes in the pre-motor window for individual note

    Parameters
    ----------
    ClusterInfo : class
    song_note : str
        notes to be used for analysis
    save_path : path
    context_selection : str
        select data from a certain context ('U' or 'D')
    npy_update : bool
        make new .npy file
    Returns
    -------
    pre_motor_spk_dict : dict
    """
    # Get number of spikes from pre-motor window per note

    from analysis.parameters import pre_motor_win_size
    from database.load import ProjectLoader
    import numpy as np

    # Create a new database (song_syllable)
    db = ProjectLoader().load_db()
    with open('database/create_syllable.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())

    cluster_id = int(ClusterInfo.name.split('-')[0])

    # Set save file (.npy)
    npy_name = ClusterInfo.name + '.npy'
    npy_name = save_path / npy_name

    if npy_name.exists() and not npy_update:
        pre_motor_spk_dict = np.load(npy_name,
                                     allow_pickle=True).item()  # all pre-deafening data to be combined for being used as a template
    else:
        nb_pre_motor_spk = np.array([], dtype=np.int)
        note_onset_ts = np.array([], dtype=np.float32)
        notes_all = ''
        contexts_all = ''

        for onsets, notes, contexts, spks in zip(ClusterInfo.onsets, ClusterInfo.syllables, ClusterInfo.contexts,
                                       ClusterInfo.spk_ts):  # loop through files
            onsets = np.delete(onsets, np.where(onsets == '*'))
            onsets = np.asarray(list(map(float, onsets)))
            notes = notes.replace('*', '')
            contexts = contexts * len(notes)

            for onset, note, context in zip(onsets, notes, contexts):  # loop through notes
                if note in song_note:
                    pre_motor_win = [onset - pre_motor_win_size, onset]
                    nb_spk = len(spks[np.where((spks >= pre_motor_win[0]) & (spks <= pre_motor_win[-1]))])
                    nb_pre_motor_spk = np.append(nb_pre_motor_spk, nb_spk)
                    note_onset_ts = np.append(note_onset_ts, onset)
                    notes_all += note
                    contexts_all += context
                    # Update database
                    # query = "INSERT INTO motif_syllable(clusterID, note, nb_premotor_spk) VALUES({}, {}, {})".format(cluster_id, note, nb_spk)
                    # query = "INSERT INTO motif_syllable(clusterID, note, nbPremotorSpk) VALUES(?, ?, ?)"
                    # db.cur.execute(query, (cluster_id, note, nb_spk))
        # db.conn.commit()
        # Store info in a dictionary
        pre_motor_spk_dict = {}
        pre_motor_spk_dict['pre_motor_win'] = pre_motor_win_size  # pre-motor window before syllable onset in ms

        for note in unique(notes_all):
            ind = find_str(notes_all, note)
            pre_motor_spk_dict[note] = {}  # nested dictionary
            pre_motor_spk_dict[note]['nb_spk'] = nb_pre_motor_spk[ind]
            pre_motor_spk_dict[note]['onset_ts'] = note_onset_ts[ind]
            pre_motor_spk_dict[note]['context'] = ''.join(np.asarray(list(contexts_all))[ind])

        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'PSD_similarity' + '/' + 'SpkCount',
                                  add_date=False)
        npy_name = ClusterInfo.name + '.npy'
        npy_name = save_path / npy_name
        np.save(npy_name, pre_motor_spk_dict)

    # Select context
    if context_selection:  # 'U' or 'D' and not None

        for note in list(pre_motor_spk_dict.keys()):
            if note is not 'pre_motor_win':
                context_arr = np.array(list(pre_motor_spk_dict[note]['context']))
                ind = np.where(context_arr == context_selection)[0]
                pre_motor_spk_dict[note]['nb_spk'] = pre_motor_spk_dict[note]['nb_spk'][ind]
                pre_motor_spk_dict[note]['onset_ts'] = pre_motor_spk_dict[note]['onset_ts'][ind]
                pre_motor_spk_dict[note]['context'] = ''.join(context_arr[ind])

    return pre_motor_spk_dict

def get_spectral_entropy(psd_array, normalize=True):

    import numpy as np

    psd_norm = psd_array / psd_array.sum(axis=0)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=0)
    if normalize:
        se /= np.log2(psd_norm.shape[1])
    return se