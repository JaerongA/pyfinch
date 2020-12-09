"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background (raw neural trace)
"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
import numpy as np
from song.analysis import *
from spike.parameters import *
from spike.load import *
from pathlib import Path
from utilities.functions import *


def get_cluster(database):

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
    cluster_path = project_path / database['birdID'] / database['taskName'] / cluster_taskSession / database[
        'site'][-2:] / 'Songs'
    cluster_path = Path(cluster_path)

    return cluster_name, cluster_path



def get_event_info(cell_path):
    """
    Obtain event info & serialized timestamps for song & neural analysis
    """
    import numpy as np
    from scipy.io import wavfile

    # List audio files
    audio_files = list(cell_path.glob('*.wav'))

    # Initialize variables
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


def get_isi(spk_ts: list):
    """Get inter-spike interval of spikes"""
    isi = []
    for spk_ts in spk_ts:
        isi.append(np.diff(spk_ts))
    return isi



class ClusterInfo:

    def __init__(self, database):

        # Set all database fields as attributes
        for col in database.keys():
            # dic[col] = database[col]
            setattr(self, col, database[col])

        # Get cluster name & path
        self.name, self.path = get_cluster(database)
        print('')
        print('Load cluster {self.name}'.format(self=self))


    # def __del__(self):
    #     print('Delete cluster : {self.name}'.format(self=self))

    def __repr__(self):
        '''Print out the name'''
        return '{self.name}'.format(self=self)

    def list_files(self):
        files = [file.stem for file in self.path.rglob('*.wav')]
        return files

    def load_events(self):
        """
        Obtain event info & serialized timestamps for song & neural analysis
        """
        file_name = self.path / 'EventInfo.npy'

        if file_name.exists():
            event_info = np.load(file_name, allow_pickle=True).item()
        else:
            event_info = get_event_info(self.path)

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


    def get_spk_corr(self, ref_spk_list, target_spk_list, normalize=False):
        """Get spike auto- or cross-correlogram"""

        import math

        time_bin = np.arange(-spk_corr_parm['lag'], spk_corr_parm['lag'] + 1, spk_corr_parm['bin_size'])
        correlogram = {}

        spk_corr = np.array([], dtype=np.float32)

        for social_context in set(self.contexts):
            # Compute spk correlogram

            corr_temp = np.zeros(len(time_bin))

            for ref_spks, target_spks, context in zip(ref_spk_list, target_spk_list, self.contexts):

                if context == social_context:
                    for ref_spk in ref_spks:
                        for target_spk in target_spks:
                            diff = target_spk - ref_spk

                            if (diff) and (diff <= spk_corr_parm['lag'] and diff >= -spk_corr_parm['lag']):
                                if diff < 0:
                                    ind = np.where(time_bin == -math.ceil(abs(diff)))
                                elif diff > 0:
                                    ind = np.where(time_bin == math.ceil((diff)))
                                # print(diff, time_bin[ind])
                                corr_temp[ind] += 1

                    # Make sure the array is symmetrical
                    first_half = np.fliplr([corr_temp[:spk_corr_parm['lag']]])[0]
                    second_half = corr_temp[spk_corr_parm['lag'] + 1:]
                    assert np.sum(first_half - second_half) == 0

                    # Normalize correlogram by the total sum (probability density conversion)
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
        """Open the directory in win explorer"""
        import webbrowser
        webbrowser.open(self.path)



class MotifInfo(ClusterInfo):

    def __init__(self, database):
        super().__init__(database)
        self.load_events()
        self.load_spk()

        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
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

                motif_spk = spks[np.where((spks >= motif_onset) & (spks <= motif_offset))]
                onsets_in_motif = onsets[start_ind:stop_ind + 1]
                offsets_in_motif = offsets[start_ind:stop_ind + 1]

                # onsets_in_motif = [onset for onset in onsets if onset != '*' and motif_onset <= float(onset)  <= motif_offset]
                # offsets_in_motif = [offset for offset in offsets if offset != '*' and motif_onset <= float(offset)  <= motif_offset]

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
            'durations': duration_list,
            'syllables': syllable_list,
            'contexts': context_list
        }

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])


    def get_peth(self):
        """Get peri-event time histograms & rasters"""
        pass




class BoutInfo(ClusterInfo):
    pass


class BaselineInfo(ClusterInfo):

    def __init__(self, database):
        super().__init__(database)
        # self.load_events()
        # self.load_spk()

        file_name = self.path / 'BaselineInfo.npy'

        if file_name.exists():
            baseline_info = np.load(file_name, allow_pickle=True).item()
        else:
            self.load_events()
            self.load_spk()

            # Store values in these lists
            file_list = []
            spk_list = []
            nb_spk_list = []
            duration_list = []
            baseline_info = {}

            list_zip = zip(self.files, self.spk_ts, self.file_start, self.onsets, self.offsets, self.syllables)

            for file, spks, file_start, onsets, offsets, syllables in list_zip:

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
                    duration_list.append((baseline_offset - baseline_onset) / 1E3)  # convert to seconds for calculating in Hz

            # set values into the database

            self.baselineFR = sum(nb_spk_list) / sum(duration_list)

            baseline_info = {
                'files': file_list,
                'spk_ts': spk_list,
                'nb_spk': nb_spk_list,
                'durations': duration_list,
                'parameter':baseline
            }
            # Save baseline_info as a numpy object
            np.save(file_name, baseline_info)

        # Set the dictionary values to class attributes
        for key in baseline_info:
            setattr(self, key, baseline_info[key])

    @property
    def isi(self):
        isi = get_isi(self.spk_ts)
        return isi















class AudioData():
    pass




class RawData(ClusterInfo):
    def __init__(self, database):
        super().__init__(database)

    def load_audio(self):
        pass

    def load_neural_trace(self, channel):

        # List .rhd files
        rhd_files = list(self.path.glob('*.rhd'))

        # Initialize variables
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

