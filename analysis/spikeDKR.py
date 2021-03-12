"""
By Jaerong
main package for neural analysis
"""

from pathlib import Path

from analysis.functionsDKR import *
from analysis.load import *
from analysis.parameters import *
from database.load import ProjectLoader
from util.functions import *
from util.spect import *


def load_song(dir):
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
    song_info = {
        'files': file_list,
        'file_start': file_start_list,
        'file_end': file_end_list,
        'onsets': onset_list,
        'offsets': offset_list,
        'durations': duration_list,
        'syllables': syllable_list,
        'contexts': context_list
    }
    return song_info


def load_audio(dir, format='wav'):
    """
    Load and concatenate all audio files (e.g., .wav) in the input dir (path)
    """
    from scipy.io import wavfile

    # List all audio files in the dir
    files = list_files(dir, format)

    # Initialize
    timestamp_concat = np.array([], dtype=np.float64)
    data_concat = np.array([], dtype=np.float64)

    # Store values in these lists
    file_list = []
    syllable_list = []
    context_list = []

    # Loop through audio files
    for file in files:

        # Load data file
        print('Loading... ' + file.stem)
        sample_rate, data = wavfile.read(file)  # note that the timestamp is in second

        # Add timestamp info
        length = data.shape[0] / sample_rate
        data_concat = np.append(data_concat, data)

        # Store results
        file_list.append(file)

    # Create timestamps
    timestamp_concat = np.arange(0, data_concat.shape[0] / sample_rate, (1 / sample_rate)) * 1E3

    # Organize data into a dictionary
    audio_info = {
        'files': file_list,
        'timestamp': timestamp_concat,
        'data': data_concat,
        'sample_rate': sample_rate
    }
    file_name = dir / "AudioData.npy"
    np.save(file_name, audio_info)

    return audio_info


def get_isi(spk_ts: list):
    """Get inter-analysis interval of spikes"""
    isi = []
    for spk_ts in spk_ts:
        isi.append(np.diff(spk_ts))
    return isi


def get_peth(evt_ts: list, spk_ts: list, *duration: float):
    """Get peri-event histogram & firing rates

    for song peth event_ts indicates syllable onset
    """

    import math

    peth = np.zeros((len(evt_ts), peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of trials x nb of time bins

    for trial_ind, (evt_ts, spk_ts) in enumerate(zip(evt_ts, spk_ts)):

        evt_ts = np.asarray(list(map(float, evt_ts))) - peth_parm['buffer']
        spk_ts -= evt_ts[0]

        for spk in spk_ts:
            ind = math.ceil(spk / peth_parm['bin_size'])
            # print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
            peth[trial_ind, ind] += 1

    time_bin = peth_parm['time_bin'] - peth_parm['buffer']

    # Truncate the array leaving out only the portion of our interest
    if duration:
        ind = (((0 - peth_parm['buffer']) <= time_bin) & (time_bin <= duration))
        peth = peth[:, ind]
        time_bin = time_bin[ind]

    return peth, time_bin


def get_pcc(fr_array):
    """
    Get pairwise cross-correlation
    Args:
        fr_array: arr (trials x time_bin)

    Returns:
        pcc_dict : dict
    """
    pcc_dict = {}
    pcc_arr = np.array([])

    for ind1, fr1 in enumerate(fr_array):
        for ind2, fr2 in enumerate(fr_array):
            if ind2 > ind1:
                if np.linalg.norm((fr1 - fr1.mean()), ord=1) * np.linalg.norm((fr2 - fr2.mean()), ord=1):
                    if not np.isnan(np.corrcoef(fr1, fr2)[0, 1]):
                        pcc_arr = np.append(pcc_arr, np.corrcoef(fr1, fr2)[0, 1])  # get correlation coefficient

    pcc_dict['array'] = pcc_arr
    pcc_dict['mean'] = round(pcc_arr.mean(), 3)
    return pcc_dict


class ClusterInfo:

    name: object

    def __init__(self, path, channel_nb, unit_nb, format='rhd', *name, update=False, time_unit='ms'):

        self.path = path
        if channel_nb:  # if a neuron was recorded
            if len(str(channel_nb)) == 1:
                self.channel_nb = 'Ch' + str(channel_nb)
            elif len(str(channel_nb)) == 2:
                self.channel_nb = 'Ch' + str(channel_nb)
        else:
            self.channel_nb = 'Ch'

        self.unit_nb = unit_nb
        self.format = format

        if name:
            self.name = name[0]
        else:
            self.name = self.path

        self.print_name()

        # Load events
        file_name = self.path / "ClusterInfo_{}_Cluster{}.npy".format(self.channel_nb, self.unit_nb)
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            cluster_info = load_song(self.path)
            # Save cluster_info as a numpy object
            np.save(file_name, cluster_info)
        else:
            cluster_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in cluster_info:
            setattr(self, key, cluster_info[key])

        # Load spike

        if channel_nb and unit_nb:
            self._load_spk(time_unit)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def print_name(self):
        print('')
        print('Load cluster {self.name}'.format(self=self))

    def list_files(self, ext: str):
        return list_files(self.path, ext)

    def _load_spk(self, time_unit, delimiter='\t'):
        """
        Load spike information
        Args:
            time_unit: unit # (in the cluster file)
            delimiter: delimiter of the cluster file (tab (\t) by default)

        Returns:
            sets spk_wf, spk_ts, nb_spk as attributes
        """

        spk_txt_file = list(self.path.glob('*' + self.channel_nb + '(merged).txt'))
        if not spk_txt_file:
            print("spk text file doesn't exist !")
            return

        spk_txt_file = spk_txt_file[0]
        spk_info = np.loadtxt(spk_txt_file, delimiter=delimiter, skiprows=1)  # skip header

        # Select only the unit (there could be multiple isolated units in the same file)
        if self.unit_nb:  # if the unit number is specified
            spk_info = spk_info[spk_info[:, 1] == self.unit_nb, :]

        spk_ts = spk_info[:, 2]  # analysis time stamps
        spk_wf = spk_info[:, 3:]  # analysis waveform
        nb_spk = spk_wf.shape[0]  # total number of spikes

        self.spk_wf = spk_wf  # individual waveforms
        self.nb_spk = nb_spk  # the number of spikes

        # Units are in second by default, but convert to  millisecond with the argument
        if time_unit is 'ms':
            spk_ts *= 1E3

        # Output analysis timestamps per file in a list
        spk_list = []
        for file_start, file_end in zip(self.file_start, self.file_end):
            spk_list.append(spk_ts[np.where((spk_ts >= file_start) & (spk_ts <= file_end))])

        self.spk_ts = spk_list  # analysis timestamps in ms
        # print("spk_ts, spk_wf, nb_spk attributes added")

    def analyze_waveform(self, interpolate=True):
        """
        Perform waveform analysis
        """

        self.avg_wf = np.nanmean(self.spk_wf, axis=0)
        self.wf_ts = np.arange(0, self.avg_wf.shape[0]) / sample_rate[self.format] * 1E3  # x-axis in ms



        def _get_spk_profile(wf_ts_interp, avg_wf_interp, interpolate=True):
            spk_height = np.abs(np.max(avg_wf) - np.min(avg_wf))  # in microseconds
            if interpolate:
                spk_width = abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1)) * ((1 / sample_rate[self.format]) / interp_factor) * 1E6  # in microseconds
            else:
                spk_width = abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1)) * (1 / sample_rate[self.format]) * 1E6  # in microseconds
            #deflection_range, half_width = get_half_width(wf_ts, avg_wf)  # get the half width from the peak deflection
            #return spk_height, spk_width, half_width, deflection_range

        if interpolate:  # interpolate the waveform to increase sampling frequency
            from scipy import interpolate

            f = interpolate.interp1d(self.wf_ts, self.avg_wf)
            wf_ts_interp = np.arange(0, self.wf_ts[-1], ((self.wf_ts[1] - self.wf_ts[0]) * (1 / interp_factor)))
            assert (np.diff(wf_ts_interp)[0] * interp_factor) == np.diff(self.wf_ts)[0]
            avg_wf_interp = f(wf_ts_interp)  # use interpolation function returned by `interp1d`

            # Replace the original value with interpolated ones
            self.wf_ts_interp = wf_ts_interp
            self.avg_wf_interp = avg_wf_interp

            deflection_range, half_width = get_half_width(wf_ts_interp, avg_wf_interp)  # get the half width from the peak deflection
            spk_height, spk_width, half_width, deflection_range = _get_spk_profile(wf_ts_interp, avg_wf_interp)
        else:
            spk_height, spk_width, half_width, deflection_range = _get_spk_profile(self.wf_ts, self.avg_wf)

        self.spk_height = round(spk_height, 3)  # in microvolts
        self.spk_width = round(spk_width, 3)  # in microseconds
        self.half_width = half_width
        self.deflection_range = deflection_range  # the range where half width was calculated

        # print("avg_wf, spk_height (uv), spk_width (us), wf_ts (ms) added")

    def get_conditional_spk(self):

        conditional_spk = {}
        conditional_spk['U'] = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'U']
        conditional_spk['D'] = [spk_ts for spk_ts, context in zip(self.spk_ts, self.contexts) if context == 'D']

        return conditional_spk

    def get_correlogram(self, ref_spk_list, target_spk_list, normalize=False):
        """Get analysis auto- or cross-correlogram"""

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

    def jitter_spk_ts(self, reproducible=False):
        """
        Add a random temporal jitter to the spike
        Parameters
        ----------
        reproducible : bool
            make the results reproducible by setting the seed as equal to index
        """
        spk_ts_jittered_list = []
        for ind, spk_ts in enumerate(self.spk_ts):
            np.random.seed()
            if reproducible:  # randomization seed
                seed = ind
                np.random.seed(seed)  # make random jitter reproducible
            else:
                seed = np.random.randint(len(self.spk_ts), size=1)
                np.random.seed(seed)  # make random jitter reproducible
            nb_spk = spk_ts.shape[0]
            jitter = np.random.uniform(-jitter_limit, jitter_limit, nb_spk)
            spk_ts_jittered_list.append(spk_ts + jitter)
        self.spk_ts_jittered = spk_ts_jittered_list

    def get_jittered_corr(self):

        from collections import defaultdict

        correlogram_jitter = defaultdict(list)

        for iter in range(shuffling_iter):
            self.jitter_spk_ts()
            corr_temp = self.get_correlogram(self.spk_ts_jittered, self.spk_ts_jittered)
            # Combine correlogram from two contexts
            for key, value in corr_temp.items():
                if key != 'parameter':
                    try:
                        correlogram_jitter[key].append(value)
                    except:
                        correlogram_jitter[key] = value

        # Convert to array
        for key, value in correlogram_jitter.items():
            correlogram_jitter[key] = (np.array(value))

        return correlogram_jitter

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

    @property
    def nb_files(self):

        nb_files = {}
        nb_files['U'] = len([context for context in self.contexts if context == 'U'])
        nb_files['D'] = len([context for context in self.contexts if context == 'D'])
        nb_files['All'] = nb_files['U'] + nb_files['D']

        return nb_files

    def nb_bouts(self, song_note):

        nb_bouts = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_bouts['U'] = get_nb_bouts(song_note, syllables)

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_bouts['D'] = get_nb_bouts(song_note, syllables)
        nb_bouts['All'] = nb_bouts['U'] + nb_bouts['D']

        return nb_bouts

    def nb_motifs(self, motif):

        nb_motifs = {}
        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'U']
        syllables = ''.join(syllable_list)
        nb_motifs['U'] = len(find_str(syllables, motif))

        syllable_list = [syllable for syllable, context in zip(self.syllables, self.contexts) if context == 'D']
        syllables = ''.join(syllable_list)
        nb_motifs['D'] = len(find_str(syllables, motif))

        nb_motifs['All'] = nb_motifs['U'] + nb_motifs['D']

        return nb_motifs

    @property
    def open_folder(self):
        open_folder(self.path)


class MotifInfo(ClusterInfo):
    """Child class of ClusterInfo"""

    def __init__(self, path, channel_nb, unit_nb, motif, format='rhd', *name, update=False):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        self.motif = motif

        if name:
           self.name = name[0]
        else:
           self.name = str(self.path)

        # Load motif info
        file_name = self.path / "MotifInfo_{}_Cluster{}.npy".format(self.channel_nb, self.unit_nb)
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            motif_info = self.load_motif()
            # Save info dict as a numpy object
            np.save(file_name, motif_info)
        else:
            motif_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])

    def load_motif(self):
        # Store values here
        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
        context_list = []

        list_zip = zip(self.files, self.spk_ts, self.onsets, self.offsets, self.syllables, self.contexts)

        for file, spks, onsets, offsets, syllables, context in list_zip:
            print('Loading... ' + file)
            onsets = onsets.tolist()
            offsets = offsets.tolist()

            # Find motifs
            motif_ind = find_str(syllables, self.motif)

            # Get syllable, analysis time stamps
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
                duration_list.append(motif_offset - motif_onset)
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
            'durations': duration_list,  # this is motif durations
            'syllables': syllable_list,
            'contexts': context_list,
            'parameter': peth_parm
        }

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])
        # Get duration
        note_duration_list, median_duration_list = self.get_note_duration()
        self.note_durations = note_duration_list
        self.median_durations = median_duration_list
        motif_info['note_durations'] = note_duration_list
        motif_info['median_durations'] = median_duration_list

        # Get PLW (piecewise linear warping)
        spk_ts_warp_list = self.piecewise_linear_warping()
        # self.spk_ts_warp = spk_ts_warp_list
        motif_info['spk_ts_warp'] = spk_ts_warp_list

        return motif_info

    def print_name(self):
        print('')
        print('Load motif {self.name}'.format(self=self))

    def __len__(self):
        return len(self.files)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    @property
    def open_folder(self):
        open_folder(self.path)

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
        Performs piecewise linear warping on raw analysis timestamps
        Based on each median note and gap durations
        """
        import copy

        spk_ts_warped_list = []
        list_zip = zip(self.note_durations, self.onsets, self.offsets, self.spk_ts)

        for motif_ind, (durations, onset, offset, spk_ts) in enumerate(list_zip):  # per motif

            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))
            # Make a deep copy of spk_ts so as to make it modification won't affect the original
            spk_new = copy.deepcopy(spk_ts)

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

                # Add spikes from motif
                ind, spk_ts_temp = extract_ind(spk_ts, [timestamp[i], timestamp[i + 1]])
                spk_ts_temp = ((ratio * ((spk_ts_temp - timestamp[0]) - diff)) + origin) + timestamp[0]
                # spk_ts_new = np.append(spk_ts_new, spk_ts_temp)
                np.put(spk_new, ind, spk_ts_temp)  # replace original spk timestamps with warped timestamps

            spk_ts_warped_list.append(spk_new)
        return spk_ts_warped_list

    def get_mean_fr(self):
        "Mean motif firing rates"
        fr_dict = {}
        motif_spk_list = []
        list_zip = zip(self.onsets, self.offsets, self.spk_ts)

        # Make sure spikes from the pre-motif buffer is not included in calculation
        for onset, offset, spks in list_zip:
            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))
            motif_spk_list.append(spks[np.where((spks >= onset[0]) & (spks <= offset[-1]))])

        for context1 in unique(self.contexts):
            nb_spk = sum([len(spk) for spk, context2 in zip(motif_spk_list, self.contexts) if context2 == context1])
            total_duration = sum(
                [duration for duration, context2 in zip(self.durations, self.contexts) if context2 == context1])
            mean_fr = nb_spk / (total_duration / 1E3)
            fr_dict[context1] = round(mean_fr, 3)
        # print("mean_fr added")
        self.mean_fr = fr_dict

    def get_peth(self, time_warp=True):
        """Get peri-event time histograms & rasters during song motif"""
        peth_dict = {}
        if time_warp:  # peth calculated from time-warped spikes by default
            # peth, time_bin = get_peth(self.onsets, self.spk_ts_warp, self.median_durations.sum())  # truncated version to fit the motif duration
            peth, time_bin = get_peth(self.onsets, self.spk_ts_warp)
        else:
            # peth, time_bin = get_peth(self.onsets, self.spk_ts, self.median_durations.sum())
            peth, time_bin = get_peth(self.onsets, self.spk_ts)

        peth_dict['peth'] = peth
        peth_dict['time_bin'] = time_bin
        peth_dict['contexts'] = self.contexts
        peth_dict['median_duration'] = self.median_durations.sum()
        return PethInfo(peth_dict)  # return peth class object for further analysis


class PethInfo():
    def __init__(self, peth_dict):
        """
        Args:
            peth_dict : dict
                    "peth" : array  (nb of trials (motifs) x time bins)
                    numbers indicate analysis counts in that bin
                "contexts" : list of strings
                    social contexts
        """

        # Set the dictionary values to class attributes
        for key in peth_dict:
            setattr(self, key, peth_dict[key])

        # Get conditional peth, fr, spike counts
        peth_dict = {}
        peth_dict['All'] = self.peth
        for context in unique(self.contexts):
            ind = np.array(self.contexts) == context
            peth_dict[context] = self.peth[ind, :]
        self.peth = peth_dict

    def get_fr(self, smoothing=True, norm_method=None, norm_factor=None):
        """
        Get trials-by-trial firing rates by default
        Args:
            smoothing: bool
                performs gaussian smoothing on the firing rates
            norm_method: str ['sum', 'factor']
                normalization by the sum (default)
            norm_factor:  float
                (e.g., baseline firing rates).
        """
        # if duration:
        #     ind = (((0 - peth_parm['buffer']) <= time_bin) & (time_bin <= duration))
        #     peth = peth[:, ind]
        #     time_bin = time_bin[ind]

        from scipy.ndimage import gaussian_filter1d

        # Get trial-by-trial firing rates
        fr_dict = {}
        for k, v in self.peth.items():  # loop through different conditions in peth dict
            fr = v / (peth_parm['bin_size'] / 1E3)  # in Hz

            if smoothing:  # Gaussian smoothing
                fr = gaussian_filter1d(fr, gauss_std)

            # Truncate values outside the range
            ind = (((0 - peth_parm['buffer']) <= self.time_bin) & (self.time_bin <= self.median_duration))
            fr = fr[:, ind]
            fr_dict[k] = fr
        self.fr = fr_dict
        self.time_bin = self.time_bin[ind]

        # Get mean firing rates
        fr_dict = {}
        for k, v in self.fr.items():
            fr = np.mean(v, axis=0)
            if norm_method == 'sum':  # normalize by the total sum
                fr = fr / sum(fr)
            elif norm_method == 'factor':  # normalize by a normalization factor (e.g., baseline firing rates)
                fr = fr / norm_factor
            fr_dict[k] = fr
        self.mean_fr = fr_dict

    def get_pcc(self):
        "Get pairwise cross-correlation"
        pcc_dict = {}
        for k, v in self.fr.items():  # loop through different conditions in peth dict
            if k != 'All':
                pcc = get_pcc(v)
                pcc_dict[k] = pcc
        self.pcc = pcc_dict

    def get_spk_count(self):

        win_size = spk_count_parm['win_size']
        spk_count_dict = {}
        fano_factor_dict = {}
        spk_count_cv_dict = {}

        for k, v in self.peth.items():  # loop through different conditions in peth dict
            spk_arr = np.empty((v.shape[0], 0), int)  # (renditions x time bins)
            if k != 'All':  # skip all trials
                win_inc = 0
                for i in range(v.shape[1] - win_size):
                    count = v[:, i: win_size + win_inc].sum(axis=1)
                    # print(f"from {i} to {win_size + win_inc}, count = {count}")
                    spk_arr = np.append(spk_arr, np.array([count]).transpose(), axis=1)
                    win_inc += 1
                # Truncate values outside the range
                ind = (((0 - peth_parm['buffer']) <= self.time_bin) & (self.time_bin <= self.median_duration))
                spk_arr = spk_arr[:, :ind.shape[0]]

                spk_count = spk_arr.sum(axis=0)
                fano_factor = spk_arr.var(axis=0) / spk_arr.mean(
                    axis=0)  # per time window (across renditions) (renditions x time window)
                spk_count_cv = spk_count.std(axis=0) / spk_count.mean(axis=0)  # cv across time (single value)

                # store values in a dictionary
                spk_count_dict[k] = spk_count
                fano_factor_dict[k] = fano_factor
                spk_count_cv_dict[k] = round(spk_count_cv, 3)

        self.spk_count = spk_count_dict
        self.fano_factor = fano_factor_dict
        self.spk_count_cv = spk_count_cv_dict

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class BoutInfo(ClusterInfo):
    """
    Get song & spike information for a song bout
    Child class of ClusterInfo
    """

    def __init__(self, path, channel_nb, unit_nb, song_note, format='rhd', *name, update=False):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        self.song_note = song_note

        if name:
           self.name = name[0]
        else:
           self.name = str(self.path)

        # Load bout info
        file_name = self.path / "BoutInfo_{}_Cluster{}.npy".format(self.channel_nb, self.unit_nb)
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            bout_info = self.load_bouts()
            # Save info dict as a numpy object
            np.save(file_name, bout_info)
        else:
            bout_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in bout_info:
            setattr(self, key, bout_info[key])

    def print_name(self):
        print('')
        print('Load bout {self.name}'.format(self=self))

    def __len__(self):
        return len(self.files)

    def load_bouts(self):
        # Store values here
        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
        context_list = []

        list_zip = zip(self.files, self.spk_ts, self.onsets, self.offsets, self.syllables, self.contexts)

        for file, spks, onsets, offsets, syllables, context in list_zip:

            bout_ind = find_str(syllables, '*')

            for ind in range(len(bout_ind)):
                if ind == 0:
                    start_ind = 0
                else:
                    start_ind = bout_ind[ind - 1] + 1
                stop_ind = bout_ind[ind] - 1
                # breakpoint()
                bout_onset = float(onsets[start_ind])
                bout_offset = float(offsets[stop_ind])

                bout_spk = spks[np.where((spks >= bout_onset) & (spks <= bout_offset))]
                onsets_in_bout = onsets[start_ind:stop_ind + 1]  # list of bout onset timestamps
                offsets_in_bout = offsets[start_ind:stop_ind + 1]  # list of bout offset timestamps

                file_list.append(file)
                spk_list.append(bout_spk)
                duration_list.append(bout_offset - bout_onset)
                onset_list.append(onsets_in_bout)
                offset_list.append(offsets_in_bout)
                syllable_list.append(syllables[start_ind:stop_ind + 1])
                context_list.append(context)

        # Organize event-related info into a single dictionary object
        bout_info = {
            'files': file_list,
            'spk_ts': spk_list,
            'onsets': onset_list,
            'offsets': offset_list,
            'durations': duration_list,  # this is bout durations
            'syllables': syllable_list,
            'contexts': context_list,
        }

        return bout_info


class BaselineInfo(ClusterInfo):

    def __init__(self, path, channel_nb, unit_nb, format='rhd', *name, update=False):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        # Load baseline info
        file_name = self.path / "BaselineInfo_{}_Cluster{}.npy".format(self.channel_nb, self.unit_nb)
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file

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
                bout_ind_list = find_str(syllables, '*')
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

        else:
            baseline_info = np.load(file_name, allow_pickle=True).item()

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

        return correlogram  # return class object for further analysis

    def get_jittered_corr(self):

        correlogram_jitter = []

        for iter in range(shuffling_iter):
            self.jitter_spk_ts()
            corr_temp = self.get_correlogram(self.spk_ts_jittered, self.spk_ts_jittered)
            correlogram_jitter.append(corr_temp)

        return np.array(correlogram_jitter)

    @property
    def mean_fr(self):
        """Mean firing rates"""
        nb_spk = sum([len(spk_ts) for spk_ts in self.spk_ts])
        total_duration = sum(self.durations)
        mean_fr = nb_spk / total_duration
        return mean_fr

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    @property
    def isi(self):
        isi = get_isi(self.spk_ts)
        return isi


class AudioData:
    """
    Create an object that has concatenated audio signal and its timestamps
    Get all data by default; specify time range if needed
    """
    def __init__(self, path, format='.wav', update=False):

        self.path = path
        self.format = format

        file_name = self.path / "AudioData.npy"
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            audio_info = load_audio(self.path, self.format)
        else:
            audio_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in audio_info:
            setattr(self, key, audio_info[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

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

    def spectrogram(self, freq_range=[300, 8000]):
        self.spect, self.spect_freq, _ = spectrogram(self.data, self.sample_rate, freq_range=freq_range)
        self.spect_time = np.linspace(self.timestamp[0], self.timestamp[-1], self.spect.shape[1])  # timestamp for spectrogram
        # print("spect, freqbins, timebins added")

    def plot_spectrogram(self, MotifInfo):
        pass


class NeuralData:
    def __init__(self, path, channel_nb, format='rhd', update=False):

        self.path = path
        self.channel = channel_nb
        self.format = format  # format of the file (e.g., rhd), this info should be in the database

        file_name = self.path / f"NeuralData_Ch{self.channel}.npy"
        if update or not file_name.exists():  # if .npy doesn't exist or want to update the file
            data_info = self.load_neural_data()
            # Save event_info as a numpy object
        else:
            data_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in data_info:
            setattr(self, key, data_info[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def load_neural_data(self):
        """
        Load and concatenate all neural data files (e.g., .rhd) in the input dir (path)
        """
        print("")
        print("Load neural data")
        # List .rhd files
        rhd_files = list(self.path.glob('*.rhd'))

        # Initialize
        timestamp_concat = np.array([], dtype=np.float64)
        amplifier_data_concat = np.array([], dtype=np.float64)

        # Store values in these lists
        file_list = []

        # Loop through Intan .rhd files
        for file in rhd_files:

            # Load data file
            print('Loading... ' + file.stem)
            file_list.append(file)
            intan = read_rhd(file)  # note that the timestamp is in second
            # Concatenate timestamps
            intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0
            if timestamp_concat.size == 0:
                timestamp_concat = np.append(timestamp_concat, intan['t_amplifier'])
            else:
                intan['t_amplifier'] += (timestamp_concat[-1] + (1 / sample_rate[self.format]))
                timestamp_concat = np.append(timestamp_concat, intan['t_amplifier'])

            # Concatenate neural data
            for ind, ch in enumerate(intan['amplifier_channels']):
                if self.channel == int(ch['native_channel_name'][-2:]):
                    amplifier_data_concat = np.append(amplifier_data_concat, intan['amplifier_data'][ind, :])

        timestamp_concat *= 1E3  # convert to microsecond

        # Organize data into a dictionary
        data_info = {
            'files': file_list,
            'timestamp': timestamp_concat,
            'data': amplifier_data_concat,
            'sample_rate': sample_rate[self.format]
        }

        file_name = self.path / f"NeuralData_Ch{self.channel}.npy"
        np.save(file_name, data_info)

        return data_info

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

class Correlogram():
    """
    Class for correlogram analysis
    """

    def __init__(self, correlogram):

        corr_center = round(correlogram.shape[0] / 2) + 1  # center of the correlogram
        self.data = correlogram
        self.time_bin = np.arange(-spk_corr_parm['lag'],
                                  spk_corr_parm['lag'] + spk_corr_parm['bin_size'],
                                  spk_corr_parm['bin_size'])
        if self.data.sum():
            self.peak_ind = np.min(
                np.abs(np.argwhere(correlogram == np.amax(correlogram)) - corr_center)) + corr_center  # index of the peak
            self.peak_latency = self.time_bin[self.peak_ind]
            self.peak_value = self.data[self.peak_ind]
            burst_range = np.arange(corr_center - (1000 / burst_hz) - 1, corr_center + (1000 / burst_hz),
                                    dtype='int')  # burst range in the correlogram
            self.burst_index = round(self.data[burst_range].sum() / self.data.sum(), 3)
        else:
            self.peak_ind = self.peak_latency = self.peak_value = self.burst_index = np.nan

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def category(self, correlogram_jitter):
        """
        Get bursting category of a neuron based on autocorrelogram
        Parameters
        ----------
        correlogram_jitter : array
            Random time-jittered correlogram for baseline setting
        Returns
            Category of a neuron ('Bursting' or 'Nonbursting')
        -------
        """
        corr_mean = correlogram_jitter.mean(axis=0)

        if corr_mean.sum():

            corr_std = correlogram_jitter.std(axis=0)
            upper_lim = corr_mean + (corr_std * 2)
            lower_lim = corr_mean - (corr_std * 2)

            self.baseline = upper_lim

            # Check peak significance
            if self.peak_value > upper_lim[self.peak_ind] and self.peak_latency <= corr_burst_crit:
                self.category = 'Bursting'
            else:
                self.category = 'NonBursting'

        else:
            self.baseline = self.category = np.array(np.nan)
        return self.category

    def plot_corr(self, ax, time_bin, correlogram,
                  title,
                  font_size=10,
                  peak_line_width=0.8,
                  normalize=False,
                  peak_line=True,
                  baseline=True):
        """
        Plot correlogram
        Parameters
        ----------
        ax : axis to plot the figure
        time_bin : array
        correlogram : array
        title : str
        font_size : title font size
        normalize : normalize the correlogram
        """
        import matplotlib.pyplot as plt
        from util.draw import remove_right_top
        if correlogram.sum():
            ax.bar(time_bin, correlogram, color='k')
            ymax = max([self.baseline.max(), correlogram.max()])
            ymax = myround(ymax, base=10)
            ax.set_ylim(0, ymax)
            plt.yticks([0, ax.get_ylim()[1]], [str(0), str(int(ymax))])
            ax.set_title(title, size=font_size)
            ax.set_xlabel('Time (ms)')
            if normalize:
                ax.set_ylabel('Prob')
            else:
                ax.set_ylabel('Count')
            remove_right_top(ax)

            if peak_line and not np.isnan(self.peak_ind):
                # peak_time_ind = np.where(self.time_bin == self.peak_latency)
                ax.axvline(x=self.time_bin[self.peak_ind], color='r', linewidth=peak_line_width, ls='--')

            if baseline and not np.isnan(self.baseline.mean()):
                ax.plot(self.time_bin, self.baseline, 'm', lw=0.5, ls='--')
        else:
            ax.axis('off')
            ax.set_title(title, size=font_size)


class BurstingInfo:

    def __init__(self, ClassInfo, *input_context):

        # ClassInfo can be BaselineInfo, MotifInfo etc
        if input_context:  # select data based on social context
            spk_list = [spk_ts for spk_ts, context in zip(ClassInfo.spk_ts, ClassInfo.contexts) if
                        context == input_context[0]]
            duration_list = [duration for duration, context in zip(ClassInfo.durations, ClassInfo.contexts) if
                             context == input_context[0]]
            self.context = input_context
        else:
            spk_list = ClassInfo.spk_ts
            duration_list = ClassInfo.durations

        # Bursting analysis
        burst_spk_list = []
        burst_duration_arr = []

        nb_bursts = []
        nb_burst_spk_list = []

        for ind, spks in enumerate(spk_list):

            # spk = bi.spk_ts[8]
            isi = np.diff(spks)  # inter-spike interval
            inst_fr = 1E3 / np.diff(spks)  # instantaneous firing rates (Hz)
            bursts = np.where(inst_fr >= burst_hz)[0]  # burst index

            # Skip if no bursting detected
            if not bursts.size:
                continue

            # Get the number of bursts
            temp = np.diff(bursts)[np.where(np.diff(bursts) == 1)].size  # check if the spikes occur in bursting
            nb_bursts = np.append(nb_bursts, bursts.size - temp)

            # Get burst onset
            temp = np.where(np.diff(bursts) == 1)[0]
            spk_ind = temp + 1
            # Remove consecutive spikes in a burst and just get burst onset

            burst_onset_ind = bursts

            for i, ind in enumerate(temp):
                burst_spk_ind = spk_ind[spk_ind.size - 1 - i]
                burst_onset_ind = np.delete(burst_onset_ind, burst_spk_ind)

            # Get burst offset index
            burst_offset_ind = np.array([], dtype=np.int)

            for i in range(bursts.size - 1):
                if bursts[i + 1] - bursts[i] > 1:  # if not successive spikes
                    burst_offset_ind = np.append(burst_offset_ind, bursts[i] + 1)

            # Need to add the subsequent spike time stamp since it is not included (burst is the difference between successive spike time stamps)
            burst_offset_ind = np.append(burst_offset_ind, bursts[bursts.size - 1] + 1)

            burst_onset = spks[burst_onset_ind]
            burst_offset = spks[burst_offset_ind]
            burst_spk_list.append(spks[burst_onset_ind[0]: burst_offset_ind[0] + 1])
            burst_duration_arr = np.append(burst_duration_arr, burst_offset - burst_onset)

            # Get the number of burst spikes
            nb_burst_spks = 1  # note that it should always be greater than 1

            if nb_bursts.size:
                if bursts.size == 1:
                    nb_burst_spks = 2
                    nb_burst_spk_list.append(nb_burst_spks)

                elif bursts.size > 1:
                    for ind in range(bursts.size - 1):
                        if bursts[ind + 1] - bursts[ind] == 1:
                            nb_burst_spks += 1
                        else:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
                            nb_burst_spks = 1

                        if ind == bursts.size - 2:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
            # print(nb_burst_spk_list)
        if sum(nb_burst_spk_list):
            self.spk_list = burst_spk_list
            self.nb_burst_spk = sum(nb_burst_spk_list)
            self.fraction = round(sum(nb_burst_spk_list) / sum([len(spks) for spks in spk_list]), 3)
            self.duration = round((burst_duration_arr).sum(), 3)  # total duration
            self.freq = round(nb_bursts.sum() / sum(duration_list), 3)
            self.mean_nb_spk = round(np.array(nb_burst_spk_list).mean(), 3)
            self.mean_duration = round(burst_duration_arr.mean(), 3)  # mean duration
        else:  # no burst spike detected
            self.spk_list = np.nan
            self.nb_burst_spk = np.nan
            self.fraction = np.nan
            self.duration = np.nan
            self.freq = np.nan
            self.mean_nb_spk = np.nan
            self.mean_duration = np.nan

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

