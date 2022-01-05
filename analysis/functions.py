"""
A collection of functions used for song & neural analysis
"""

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
    if unit == 'second':
        onsets /= 1E3
        offsets /= 1E3
        intervals /= 1E3
        durations /= 1E3

    return onsets, offsets, intervals, durations, syllables, contexts


def get_note_type(syllables, song_db) -> list:
    """
    Function to determine the category of the syllable
    Parameters
    ----------
    syllables : str
    song_db : db

    Returns
    -------
    type_str : list
    """
    type_str = []
    for syllable in syllables:
        if syllable in song_db.motif:
            type_str.append('M')  # motif
        elif syllable in song_db.calls:
            type_str.append('C')  # call
        elif syllable in song_db.introNotes:
            type_str.append('I')  # intro notes
        else:
            type_str.append(None)  # intro notes
    return type_str


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
                if i != 0:
                    bout_labeling += '*' + target[ind[i - 1] + 1:ind[i] + 1]
                else:
                    bout_labeling = target[:item + 1]
            bout_labeling += '*' + target[ind[i] + 1:]

        bout_labeling += '*'  # end with an asterisk


    elif isinstance(target, np.ndarray):
        if len(ind):
            for i, item in enumerate(ind):
                if i == 0:
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
    """
    Count the number of bouts (only includes those having at least one song note)
    Parameters
    ----------
    song_note : str
        syllables that are part of a motif (e.g., abcd)
    bout_labeling : str
        syllables that are demarcated by * (bout) (e.g., iiiiiiiiabcd*jiiiabcdji*)
    Returns
    -------
    """
    nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if
                    unique_nb_notes_in_bout(song_note, bout)])
    return nb_bouts


def get_snr(avg_wf, raw_neural_trace, filter_crit=5):
    """
    Calculate signal-to-noise ratio of sorted spike relative to the background neural trace
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
    from scipy.signal import hilbert

    # Get signal envelop of the raw trace and filter out ranges that are too large (e.g., motor artifacts)
    envelope = np.abs(hilbert(raw_neural_trace))

    envelope_crit = envelope.mean() + envelope.std() * filter_crit
    waveform_crit = abs(avg_wf).mean() + abs(avg_wf).std() * filter_crit

    crit = max(envelope_crit, waveform_crit)
    raw_neural_trace[envelope > crit] = np.nan

    # plt.plot(raw_neural_trace), plt.show()
    snr = 10 * np.log10(np.nanvar(avg_wf) / np.nanvar(raw_neural_trace))  # in dB
    return round(snr, 3)


def get_half_width(wf_ts, avg_wf):
    import numpy as np

    # Find the negative (or positive if inverted) deflection
    if np.argmin(avg_wf) > np.argmax(avg_wf):  # inverted waveform (peak comes first in extra-cellular recording)
        deflection_baseline = np.abs(avg_wf).mean() + np.abs(avg_wf).std(
            axis=0)  # the waveform baseline. Finds values above the baseline
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
    half_width = np.nan

    for ind, _ in enumerate(diff_ind):
        if ind < len(diff_ind) - 1:
            range = diff_ind[ind: ind + 2]
            if range[0] < peak_ind < range[1]:
                deflection_range = range
                break
    if deflection_range:
        # sometimes half-width cannot be properly estimated unless the signal is interpolated
        # in such cases, the function will just return nan
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
            timestamp = np.round(np.linspace(0, length, data.shape[0]) * 1E3,
                                 3)  # start from t = 0 in ms, reduce floating precision
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
    import numpy as np

    psd_dict = {}
    psd_list_basis = []
    note_list_basis = []

    psd_array = np.asarray(psd_list)  # number of syllables x psd (get_basis_psd function accepts array format only)
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
            if note != 'pre_motor_win':
                context_arr = np.array(list(pre_motor_spk_dict[note]['context']))
                ind = np.where(context_arr == context_selection)[0]
                pre_motor_spk_dict[note]['nb_spk'] = pre_motor_spk_dict[note]['nb_spk'][ind]
                pre_motor_spk_dict[note]['onset_ts'] = pre_motor_spk_dict[note]['onset_ts'][ind]
                pre_motor_spk_dict[note]['context'] = ''.join(context_arr[ind])

    return pre_motor_spk_dict


def get_spectral_entropy(psd_array, normalize=True, mode=None):
    import numpy as np

    if mode == 'spectral':
        # Get time resolved version of the spectral entropy
        psd_array = psd_array.mean(axis=1)  # time-averaged spectrogram
        psd_norm = psd_array / psd_array.sum()
        se = -(psd_norm * np.log2(psd_norm)).sum()
        se /= np.log2(psd_norm.shape[0])
        return se

    elif mode == 'spectro_temporal':
        se_dict = {}
        se_array = np.array([], dtype=np.float32)
        for i in range(psd_array.shape[1]):
            psd_norm = psd_array[:, i] / psd_array[:, i].sum()
            se = -(psd_norm * np.log2(psd_norm)).sum()
            if normalize:
                se /= np.log2(psd_norm.shape[0])
            se_array = np.append(se_array, se)
        se_dict['array'] = se_array
        se_dict['mean'] = se_array.mean()
        se_dict['var'] = se_array.var()
        # se_dict['var'] = 1 / -np.log(se_array.var())
        # se_dict['var'] = se_array.std() / se_array.mean()  # calculate cv
        return se_dict


def get_ff(data, sample_rate, ff_low, ff_high, ff_harmonic=1):
    """
    Calculate fundamental frequency (FF) from the FF segment
    Parameters
    ----------
    data : array
    sample_rate : int
        data sampling rate
    ff_low : int
        Lower limit
    ff_high : int
        Upper limit
    ff_harmonic :  int (1 by default)
        harmonic detection
    Returns
    -------
    ff : float
    """
    from scipy.signal import find_peaks
    import statsmodels.tsa.stattools as smt
    import matplotlib.pyplot as plt
    import numpy as np
    from util.functions import para_interp

    # Get peak of the auto-correlogram
    corr = smt.ccf(data, data, adjusted=False)
    corr_win = corr[3: round(sample_rate / ff_low)]
    peak_ind, property = find_peaks(corr_win, height=0)

    # Plot auto-correlation (for debugging)
    # plt.plot(corr_win)
    # plt.plot(peak_ind, corr_win[peak_ind], "x")
    # plt.show()

    # Find FF
    ff_list = []
    ff = None
    # loop through the peak until FF is found in the desired range
    for ind in property['peak_heights'].argsort()[::-1]:
        if not (peak_ind[ind] == 0 or (
                peak_ind[ind] == len(corr_win))):  # if the peak is not in first and last indices
            target_peak_ind = peak_ind[ind]
            target_peak_amp = corr_win[
                              target_peak_ind - 1: target_peak_ind + 2]  # find the peak using two neighboring values using parabolic interpolation
            target_peak_ind = np.arange(target_peak_ind - 1, target_peak_ind + 2)
            peak, _ = para_interp(target_peak_ind, target_peak_amp)

            # period = peak + 3
            period = peak + (3 * ff_harmonic)
            temp_ff = round(sample_rate / period, 3)
            ff_list.append(temp_ff)

        ff = [ff for ff in ff_list if ff_low < ff < ff_high]
        if not bool(ff):  # return nan if ff is outside the range
            ff = None
        else:
            ff = ff[0] / ff_harmonic
    return ff


def normalize_from_pre(df, var_name: str, note: str):
    """Normalize post-deafening values using pre-deafening values"""
    pre_val = df.loc[(df['note'] == note) & (df['taskName'] == 'Predeafening')][var_name]
    pre_val = pre_val.mean()

    post_val = df.loc[(df['note'] == note) & (df['taskName'] == 'Postdeafening')][var_name]
    norm_val = post_val / pre_val

    return norm_val


def add_pre_normalized_col(df, col_name_to_normalize, col_name_to_add,
                           save_path=None, csv_name=None, save_csv=False):
    """Normalize relative to pre-deafening mean"""
    import numpy as np

    df[col_name_to_add] = np.nan

    bird_list = sorted(set(df['birdID'].to_list()))
    for bird in bird_list:

        temp_df = df.loc[df['birdID'] == bird]
        note_list = temp_df['note'].unique()

        for note in note_list:
            norm_val = normalize_from_pre(temp_df, col_name_to_normalize, note)
            add_ind = temp_df.loc[(temp_df['note'] == note) & (temp_df['taskName'] == 'Postdeafening')].index
            df.loc[add_ind, col_name_to_add] = norm_val

    if save_csv:
        df.to_csv(save_path / csv_name, index=False, header=True)

    return df


def get_bird_colors(birds):
    """
    Get separate colors for different birds
    Parameters
    ----------
    birds : list

    Returns
    -------
    bird_color : dict
    """
    import matplotlib.cm as cm
    import numpy as np

    x = np.arange(10)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    bird_color = {bird: color for bird, color in zip(birds, colors)}

    return bird_color


def get_spectrogram(timestamp, data, sample_rate, freq_range=[300, 8000]):
    """Calculate spectrogram"""
    import numpy as np
    from util.spect import spectrogram

    spect, spect_freq, _ = spectrogram(data, sample_rate, freq_range=freq_range)
    spect_time = np.linspace(timestamp[0], timestamp[-1], spect.shape[1])  # timestamp for spectrogram
    return spect_time, spect, spect_freq


def align_waveform(spk_wf):
    """Align spike waveforms relative to the max location of the average waveform"""
    import numpy as np

    aligned_wf = np.empty((spk_wf.shape[0], spk_wf.shape[1]))
    aligned_wf[:] = np.nan

    max_first = False  # max value (peak) comes first

    avg_wf = spk_wf.mean(axis=0)
    if np.argmin(avg_wf) < np.argmax(avg_wf):  # inverted waveform (peak comes first in extra-cellular recording)
        max_first = True

    if max_first:
        template_max_ind = np.argmin(avg_wf)
    else:
        template_max_ind = np.argmax(avg_wf)

    for ind, wf in enumerate(spk_wf):

        new_wf = np.empty((spk_wf.shape[1]))
        new_wf[:] = np.nan
        if max_first:
            max_ind = np.argmin(wf)
        else:
            max_ind = np.argmax(wf)

        if template_max_ind != max_ind:
            max_diff = max_ind - template_max_ind
            if max_diff > 0:
                new_wf[:-max_diff] = wf[max_diff:]
            else:
                new_wf[abs(max_diff):] = wf[:max_diff]
            aligned_wf[ind] = new_wf
        else:
            aligned_wf[ind] = wf
    return aligned_wf
