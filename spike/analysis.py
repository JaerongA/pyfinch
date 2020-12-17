"""
By Jaerong
A main package for neural analysis
"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from song.analysis import *
from spike.parameters import *
from spike.load import *
from pathlib import Path
from util.functions import *
from util.spect import *


def load_info(database):
    """
    Return the list of files in the current directory
        Input: SQL object (database row)
        Output: name of the cluster
    """
    cluster_id = ''
    if len(database['id']) == 1:
        cluster_id = '00' + database['id']
    elif len(database['id']) == 2:
        cluster_id = '0' + database['id']
    cluster_taskSession = ''
    if len(str(database['taskSession'])) == 1:
        cluster_taskSession = 'D0' + str(database['taskSession'])
    elif len(str(database['taskSession'])) == 2:
        cluster_taskSession = 'D' + str(database['taskSession'])
    cluster_taskSession += '(' + str(database['sessionDate']) + ')'

    cluster_name = [cluster_id, database['birdID'], database['taskName'], cluster_taskSession,
                    database['site'], database['channel'], database['unit']]
    cluster_name = '-'.join(map(str, cluster_name))

    # Get cluster path
    project_path = ProjectLoader().path
    cluster_path = project_path / database['birdID'] / database['taskName'] / cluster_taskSession / database['site'][-2:] / 'Songs'
    cluster_path = Path(cluster_path)

    return cluster_name, cluster_path


def load_events(dir):
    """
    Obtain event info & serialized timestamps for song & neural analysis
    """
    import numpy as np
    from scipy.io import wavfile

    # List audio files
    audio_files = list(dir.glob('*.wav'))

    # Initialize
    timestamp_serialized = np.array([], dtype=np.float32)

    # Store values in these lists
    file_list = []
    file_start_list = []
    file_end_list = []
    onset_list = []
    offset_list = []
    duration_list = []
    syllable_list = []
    context_list = []

    # Loop through Intan .rhd files
    for file in audio_files:

        # Load audio files
        print('Loading... ' + file.stem)
        sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
        length = data.shape[0] / sample_rate
        timestamp = np.linspace(0., length, data.shape[0]) * 1E3  # start from t = 0 in ms

        # Load the .not.mat file
        notmat_file = file.with_suffix('.wav.not.mat')
        onsets, offsets, intervals, durations, syllables, contexts = read_not_mat(notmat_file, unit='ms')

        start_ind = timestamp_serialized.size  # start of the file

        if timestamp_serialized.size:
            timestamp += (timestamp_serialized[-1] + (1 / sample_rate))
        timestamp_serialized = np.append(timestamp_serialized, timestamp)

        # File information (name, start & end timestamp of each file)
        file_list.append(file.stem)
        file_start_list.append(timestamp_serialized[start_ind])  # in ms
        file_end_list.append(timestamp_serialized[-1])  # in ms

        onsets += timestamp[0]
        offsets += timestamp[0]

        # Demarcate song bouts
        onset_list.append(demarcate_bout(onsets, intervals))
        offset_list.append(demarcate_bout(offsets, intervals))
        duration_list.append(demarcate_bout(durations, intervals))
        syllable_list.append(demarcate_bout(syllables, intervals))
        context_list.append(contexts)

    # Organize event-related info into a single dictionary object
    event_info = {
        'files': file_list,
        'file_start': file_start_list,
        'file_end': file_end_list,
        'onsets': onset_list,
        'offsets': offset_list,
        'durations': duration_list,
        'syllables': syllable_list,
        'contexts': context_list
    }
    return event_info


def load_audio(dir):
    """
    Load and concatenate all audio files (e.g., .wav) in the input dir (path)
    """
    from scipy.io import wavfile

    file_name = dir / 'AudioData.npy'

    if file_name.exists():
        audio_info = np.load(file_name, allow_pickle=True).item()
    else:

        # List all audio files in the dir
        files = list_files(dir, '.wav')

        # Initialize
        timestamp_concat = np.array([], dtype=np.float64)
        data_concat = np.array([], dtype=np.float64)

        # Store values in these lists
        file_list = []
        syllable_list = []
        context_list = []

        # Loop through audio files
        for file in files:

            sample_rate, data = wavfile.read(file)  # note that the timestamp is in second
            length = data.shape[0] / sample_rate
            timestamp = np.round(np.linspace(0, length, data.shape[0]) *1E3, 3) # start from t = 0 in ms, reduce floating precision

            start_ind = timestamp_concat.size  # start of the file

            # print('timestamp size = {}, data size = {}'.format(len(timestamp), len(data)))

            # Concatenate timestamp and data
            if timestamp_concat.size:
                timestamp += (timestamp_concat[-1] + (1 / sample_rate))
            timestamp_concat = np.append(timestamp_concat, timestamp)

            data_concat = np.append(data_concat, data)

            # Store results
            file_list.append(file)

        # Organize data into a dictionary
        audio_info = {
            'files': file_list,
            'timestamp': timestamp_concat,
            'data': data_concat,
            'sample_rate': sample_rate
        }
        np.save(file_name, audio_info)

    return audio_info


def get_isi(spk_ts: list):
    """Get inter-spike interval of spikes"""
    isi = []
    for spk_ts in spk_ts:
        isi.append(np.diff(spk_ts))
    return isi


def get_peth(evt_ts_list, spk_ts_list, *cond_list):
    """Get peri-event histogram & firing rates"""

    import math

    peth = np.zeros((len(evt_ts_list), peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of trials x nb of time bins

    for trial_ind, (evt_ts, spk_ts) in enumerate(zip(evt_ts_list, spk_ts_list)):

        evt_ts = np.asarray(list(map(float, evt_ts)))
        spk_ts -= evt_ts[0]

        for spk in spk_ts:
            ind = math.ceil(spk / peth_parm['bin_size'])
            # print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
            peth[trial_ind, ind] += 1
    return peth


class ClusterInfo:
    ##TODO: incorporate load_spk, load_events in __init__
    def __init__(self, database):

        # Set all database fields as attributes
        for col in database.keys():
            # dic[col] = database[col]
            setattr(self, col, database[col])

        # Get cluster name & path
        self.name, self.path = load_info(database)
        print('')
        print('Load cluster {self.name}'.format(self=self))

    # def __del__(self):
    #     print('Delete cluster : {self.name}'.format(self=self))

    def __repr__(self):
        '''Print out the name'''
        return '{self.name}'.format(self=self)

    def list_files(self, ext: str):
        return list_files(self.path, ext)

    def load_events(self):
        """
        Obtain event info & serialized timestamps for song & neural analysis
        """
        file_name = self.path / 'EventInfo.npy'

        if file_name.exists():
            event_info = np.load(file_name, allow_pickle=True).item()
        else:
            event_info = load_events(self.path)

            # Save event_info as a numpy object
            np.save(file_name, event_info)

        # Set the dictionary values to class attributes
        for key in event_info:
            setattr(self, key, event_info[key])

        print("files, file_start, file_end, onsets, offsets, durations, syllables, contexts attributes added")

    def load_spk(self, unit='ms', update=False):

        spk_txt_file = list(self.path.glob('*' + self.channel + '(merged).txt'))[0]
        spk_info = np.loadtxt(spk_txt_file, delimiter='\t', skiprows=1)  # skip header
        unit_nb = int(self.unit[-2:])

        # Select only the unit (there could be multiple isolated units in the same file)
        if unit_nb:  # if the unit number is specified
            spk_info = spk_info[spk_info[:, 1] == unit_nb, :]

        spk_ts = spk_info[:, 2]  # spike time stamps
        spk_wf = spk_info[:, 3:]  # spike waveform
        nb_spk = spk_wf.shape[0]  # total number of spikes

        self.spk_wf = spk_wf  # individual waveforms
        self.nb_spk = nb_spk  # the number of spikes

        # Units are in second by default, but convert to  millisecond with the argument
        if unit is 'ms':
            spk_ts *= 1E3

        # Output spike timestamps per file in a list
        spk_list = []
        for file_start, file_end in zip(self.file_start, self.file_end):
            spk_list.append(spk_ts[np.where((spk_ts >= file_start) & (spk_ts <= file_end))])

        self.spk_ts = spk_list  # spike timestamps in ms
        print("spk_ts, spk_wf, nb_spk attributes added")

    def analyze_waveform(self):
        # Conduct waveform analysis
        if not hasattr(self, 'avg_wf'):
            print("waveform not loaded - run 'load_spk()' first!")
        else:
            avg_wf = np.nanmean(self.spk_wf, axis=0)
            spk_height = np.abs(np.max(avg_wf) - np.min(avg_wf))  # in microseconds
            spk_width = abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1)) * (
                    1 / sample_rate[self.format]) * 1E6  # in microseconds

            self.avg_wf = avg_wf
            self.spk_height = spk_height  # in microvolts
            self.spk_width = spk_width  # in microseconds
            print("avg_wf, spk_height, spk_width added")

    def get_conditional_spk(self):

        conditional_spk = {}
        conditional_spk['U'] = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'U']
        conditional_spk['D'] = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'D']

        return conditional_spk

    def get_correlogram(self, ref_spk_list, target_spk_list, normalize=False):
        """Get spike auto- or cross-correlogram"""

        import math

        # time_bin = np.arange(-spk_corr_parm['lag'], spk_corr_parm['lag'] + 1, spk_corr_parm['bin_size'])
        correlogram = {}

        # spk_corr = np.array([], dtype=np.float32)

        for social_context in set(self.contexts):
            # Compute spk correlogram

            corr_temp = np.zeros(len(spk_corr_parm['time_bin']))

            for ref_spks, target_spks, context in zip(ref_spk_list, target_spk_list, self.contexts):

                if context == social_context:
                    for ref_spk in ref_spks:
                        for target_spk in target_spks:
                            diff = target_spk - ref_spk  # time difference between two spikes
                            if (diff) and (diff <= spk_corr_parm['lag'] and diff >= -spk_corr_parm['lag']):
                                if diff < 0:
                                    ind = np.where(spk_corr_parm['time_bin'] <= -math.ceil(abs(diff)))[0][-1]
                                elif diff > 0:
                                    ind = np.where(spk_corr_parm['time_bin'] >= math.ceil(diff))[0][0]
                                # print("diff = {}, bin index = {}".format(diff, spk_corr_parm['time_bin'][ind]))  # for debugging
                                corr_temp[ind] += 1

                    # Make sure the array is symmetrical
                    first_half = np.fliplr([corr_temp[:int((spk_corr_parm['lag'] / spk_corr_parm['bin_size']))]])[0]
                    second_half = corr_temp[int((spk_corr_parm['lag'] / spk_corr_parm['bin_size'])) + 1:]
                    assert np.sum(first_half - second_half) == 0

                    # Normalize correlogram by the total sum (convert to probability density )
                    if normalize:
                        corr_temp /= np.sum(correlogram)

            correlogram[social_context] = corr_temp
        correlogram['parameter'] = spk_corr_parm  # store parameters in the dictionary

        return correlogram

    def get_isi(self):

        isi = {}
        spk_list = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'U']
        isi['U'] = get_isi(spk_list)
        spk_list = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'D']
        isi['D'] = get_isi(spk_list)
        return isi

    @classmethod
    def plot_isi(isi):
        pass

    def get_fr(self):
        pass

    def bursting_analysis(self):
        pass

    @property
    def nb_files(self):

        nb_files = {}

        nb_files['U'] = len([context for context in self.contexts if context == 'U'])
        nb_files['D'] = len([context for context in self.contexts if context == 'D'])
        nb_files['ALL'] = nb_files['U'] + nb_files['D']

        return nb_files

    @property
    def nb_bouts(self):

        nb_bouts = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.context) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_bouts['U'] = get_nb_bouts(self.songNote, syllables)

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.context) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_bouts['D'] = get_nb_bouts(self.songNote, syllables)
        nb_bouts['ALL'] = nb_bouts['U'] + nb_bouts['D']

        return nb_bouts

    @property
    def nb_motifs(self):

        nb_motifs = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_motifs['U'] = len(find_str(self.motif, syllables))

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_motifs['D'] = len(find_str(self.motif, syllables))

        nb_motifs['ALL'] = nb_motifs['U'] + nb_motifs['D']

        return nb_motifs

    @property
    def open_folder(self):
        open_folder(self.path)


class MotifInfo(ClusterInfo):

    def __init__(self, database, update=False):
        super().__init__(database)
        # Load parent attributes but this will be overwritten
        self.load_events()
        self.load_spk()

        file_name = self.path / 'MotifInfo.npy'

        if not update:  # file already exists or you don't want to update the file
            baseline_info = np.load(file_name, allow_pickle=True).item()
        else:  # create a new file

            # Store values here
            file_list = []
            spk_list = []
            onset_list = []
            offset_list = []
            syllable_list = []
            motif_duration_list = []
            context_list = []
            motif_info = {}

            list_zip = zip(self.files, self.spk_ts, self.onsets, self.offsets, self.syllables, self.contexts)

            for file, spks, onsets, offsets, syllables, context in list_zip:

                onsets = onsets.tolist()
                offsets = offsets.tolist()

                # Find motifs
                motif_ind = find_str(self.motif, syllables)

                # Get syllable, spike time stamps
                for ind in motif_ind:
                    # start (first syllable) and stop (last syllable) index of a motif
                    start_ind = ind
                    stop_ind = ind + len(self.motif) - 1

                    motif_onset = float(onsets[start_ind])
                    motif_offset = float(offsets[stop_ind])

                    motif_spk = spks[np.where((spks >= motif_onset - peth_parm['buffer']) & (spks <= motif_offset))]
                    onsets_in_motif = onsets[start_ind:stop_ind + 1]  # list of motif onset timestamps
                    offsets_in_motif = offsets[start_ind:stop_ind + 1]  # list of motif offset timestamps

                    file_list.append(file)
                    spk_list.append(motif_spk)
                    motif_duration_list.append(motif_offset - motif_onset)
                    onset_list.append(onsets_in_motif)
                    offset_list.append(offsets_in_motif)
                    syllable_list.append(syllables[start_ind:stop_ind + 1])
                    context_list.append(context)

            # Organize event-related info into a single dictionary object
            motif_info = {
                'files': file_list,
                'spk_ts': spk_list,
                'onsets': onset_list,
                'offsets': offset_list,
                'motif_durations': motif_duration_list,
                'syllables': syllable_list,
                'contexts': context_list
            }

            # Set the dictionary values to class attributes
            for key in motif_info:
                setattr(self, key, motif_info[key])

            # Get duration
            note_durations, median_durations = self.get_note_duration()
            motif_info['note_durations'] = note_durations
            motif_info['median_durations'] = median_durations
            self.note_durations = note_durations
            self.median_durations = median_durations

            # Get PLW (piecewise linear warping)
            spk_ts_warp_list = self.piecewise_linear_warping()
            motif_info['spk_ts_warp'] = spk_ts_warp_list
            self.spk_ts_warp = spk_ts_warp_list

            # Save baseline_info as a numpy object
            np.save(file_name, motif_info)


    def __len__(self):
        return len(self.files)


    def get_note_duration(self):
        # Calculate note & gap duration per motif
        note_durations = np.empty((len(self), len(self.motif) * 2 - 1))

        list_zip = zip(self.onsets, self.offsets)

        for motif_ind, (onset, offset) in enumerate(list_zip):

            # Convert from string to array of floats
            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))

            # Calculate note & interval duration
            timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
            timestamp = sum(timestamp, [])

            for i in range(len(timestamp) - 1):
                note_durations[motif_ind, i] = timestamp[i + 1] - timestamp[i]

        # Get median duration
        median_durations = np.median(note_durations, axis=0)

        return note_durations, median_durations


    def piecewise_linear_warping(self):
        """
        Performs piecewise linear warping on raw spike timestamps
        Based on each median note and gap durations
        """

        spk_ts_warped_list = []

        list_zip = zip(self.note_durations, self.onsets, self.offsets, self.spk_ts)

        for motif_ind, (durations, onset, offset, spk_ts) in enumerate(list_zip):  # per motif

            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))

            # Calculate note & interval duration
            timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
            timestamp = sum(timestamp, [])

            for i in range(0, len(self.median_durations)):
                ratio = self.median_durations[i] / durations[i]
                diff = timestamp[i] - timestamp[0]
                if i == 0:
                    origin = 0
                else:
                    origin = sum(self.median_durations[:i])
                ind, spk_ts_new = extract(spk_ts, [timestamp[i], timestamp[i + 1]])
                spk_ts_new = ((ratio * ((spk_ts_new - timestamp[0]) - diff)) + origin) + timestamp[0]
                np.put(spk_ts, ind, spk_ts_new)  # replace original spk timestamps with warped timestamps

            spk_ts_warped_list.append(spk_ts)
        return spk_ts_warped_list


    def get_peth(self, warped=True):
        """Get peri-event time histograms & rasters during song motif"""
        peth_dict = {}

        if warped:  # peth calculated from time-warped spikes by default
            peth = get_peth(self.onsets, self.spk_ts_warp)
        else:
            peth = get_peth(self.onsets, self.spk_ts)

        peth_dict['peth'] = peth
        peth_dict['contexts'] = self.contexts
        return PethInfo(peth_dict)  # return peth class object for further analysis


class PethInfo():
    def __init__(self, peth_dict):
        """
        Args:
            peth_dict : dict
                "peth" : array  (nb of trials (motifs) x time bins)
                    numbers indicate spike counts in that bin
                "contexts" : list of strings
                    social contexts
        """

        # Set the dictionary values to class attributes
        for key in peth_dict:
            setattr(self, key, peth_dict[key])

        self.peth = peth_dict['peth']
        self.count = sum(self.peth)  # sum number of spikes per time bin
        self.fr = self.count / peth_parm['bin_size']
        self.time_bin = peth_parm['time_bin'] - peth_parm['buffer']
        ##TODO: add self.fr, normalize, plot option

    def __len__(self):
        return self.peth.shape[0]

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class BoutInfo(ClusterInfo):
    pass


class BaselineInfo(ClusterInfo):

    def __init__(self, database, update=False):
        super().__init__(database)
        # Load parent attributes but this will be overwritten
        self.load_events()
        self.load_spk()

        file_name = self.path / 'BaselineInfo.npy'

        if not update:  # file already exists or you don't want to update the file
            baseline_info = np.load(file_name, allow_pickle=True).item()
        else:  # create a new file

            # Store values in here
            file_list = []
            spk_list = []
            nb_spk_list = []
            duration_list = []
            context_list = []
            baseline_info = {}

            list_zip = zip(self.files, self.spk_ts, self.file_start, self.onsets, self.offsets, self.syllables,
                           self.contexts)

            for file, spks, file_start, onsets, offsets, syllables, context in list_zip:

                baseline_spk = []
                bout_ind_list = find_str('*', syllables)
                bout_ind_list.insert(0, -1)  # start from the first index

                for bout_ind in bout_ind_list:
                    # print(bout_ind)
                    if bout_ind == len(syllables) - 1:  # skip if * indicates the end syllable
                        continue

                    baseline_onset = float(onsets[bout_ind + 1]) - baseline['time_buffer'] - baseline['time_win']

                    if bout_ind > 0 and baseline_onset < float(offsets[
                                                                   bout_ind - 1]):  # skip if the baseline starts before the offset of the previous syllable
                        continue

                    if baseline_onset < file_start:
                        baseline_onset = file_start

                    baseline_offset = float(onsets[bout_ind + 1]) - baseline['time_buffer']

                    if baseline_offset - baseline_onset < 0:  # skip if there's not enough baseline period at the start of a file
                        continue

                    if baseline_onset > baseline_offset:
                        print('start time ={} to end time = {}'.format(baseline_onset, baseline_offset))
                        breakpoint()

                    baseline_spk = spks[np.where((spks >= baseline_onset) & (spks <= baseline_offset))]

                    file_list.append(file)
                    spk_list.append(baseline_spk)
                    nb_spk_list.append(len(baseline_spk))
                    duration_list.append(
                        (baseline_offset - baseline_onset) / 1E3)  # convert to seconds for calculating in Hz
                    context_list.append(context)

            # set values into the database
            self.baselineFR = sum(nb_spk_list) / sum(duration_list)

            baseline_info = {
                'files': file_list,
                'spk_ts': spk_list,
                'nb_spk': nb_spk_list,
                'durations': duration_list,
                'contexts': context_list,
                'parameter': baseline
            }
            # Save baseline_info as a numpy object
            np.save(file_name, baseline_info)

        # Set the dictionary values to class attributes
        for key in baseline_info:
            setattr(self, key, baseline_info[key])

    def get_correlogram(self, ref_spk_list, target_spk_list, normalize=False):
        """Override the parent method
        combine correlogram from undir and dir since no contextual differentiation is needed in baseline"""

        correlogram_all = super().get_correlogram(ref_spk_list, target_spk_list, normalize=False)
        correlogram = np.zeros(len(spk_corr_parm['time_bin']))

        # Combine correlogram from two contexts
        for key, value in correlogram_all.items():
            if key in ['U', 'D']:
                correlogram += value
        return correlogram

    @property
    def isi(self):
        isi = get_isi(self.spk_ts)
        return isi


class AudioData():
    """
    Create an object that has concatenated audio signal and its timestamps
    Get all data by default; specify time range if needed
    """
    def __init__(self, database, update=False, ext='.wav'):
        self.name, self.path = load_info(database)
        self.files = list_files(self.path, ext)
        audio_info = load_audio(self.path)

        # Set the dictionary values to class attributes
        for key in audio_info:
            setattr(self, key, audio_info[key])

    @property
    def open_folder(self):
        open_folder(self.path)

    def extract(self, time_range):
        """
        Extracts data from the specified range
        Args:
            time_range: list

        Returns:
        """
        start = time_range[0]
        end = time_range[-1]

        ind = np.where((self.timestamp >= start) & (self.timestamp <= end))
        self.timestamp = self.timestamp[ind]
        self.data = self.data[ind]

        return self

    def spectrogram(self, freq_range):
        self.spect, self.freqbins, self.timebins = spectrogram(self.data, self.sample_rate, freq_range=freq_range)
        print("spect, freqbins, timebins added")


class NerualData(ClusterInfo):
    def __init__(self, database):
        super().__init__(database)

    def load_neural_trace(self, channel):

        # List .rhd files
        rhd_files = list(self.path.glob('*.rhd'))

        # Initialize
        amplifier_data_serialized = np.array([], dtype=np.float64)

        # Loop through Intan .rhd files
        for file in rhd_files:

            # Load the .rhd file
            intan = read_rhd(file)  # note that the timestamp is in second

            # Serialize time stamps
            intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0

            if amplifier_data_serialized.size == 0:
                amplifier_data_serialized = np.append(amplifier_data_serialized, intan['t_amplifier'])
            else:
                intan['t_amplifier'] += (amplifier_data_serialized[-1] + (1 / sample_rate[self.format]))
                amplifier_data_serialized = np.append(amplifier_data_serialized, intan['t_amplifier'])

    def load_timestamp(self):
        pass
