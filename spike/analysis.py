"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background (raw neural trace)
"""

from database.load import project
import matplotlib.pyplot as plt
import numpy as np
from song.functions import *
from spike.parameters import *
from spike.load import read_rhd
from pathlib import Path
from utilities.save import save2json


def get_event_info(cell_path):
    """
    Obtain event info & serialized timestamps for song & neural analysis
    """
    ## Todo : change .wav timestamps instead of intan

    import numpy as np

    # List .rhd files
    rhd_files = list(cell_path.glob('*.rhd'))

    # Initialize variables
    t_amplifier_serialized = np.array([], dtype=np.float64)

    # Store values in these lists
    file_list = []
    file_start_list = []
    file_end_list = []
    onset_list = []
    offset_list = []
    syllable_list = []
    context_list = []

    # Loop through Intan .rhd files
    for file in rhd_files:

        # Load the .rhd file
        print('Loading... ' + file.stem)
        intan = read_rhd(file)  # note that the timestamp is in second

        # Load the .not.mat file
        notmat_file = file.with_suffix('.wav.not.mat')
        onsets, offsets, intervals, duration, syllables, context = read_not_mat(notmat_file, unit='ms')

        # Serialize time stamps
        intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0
        start_ind = t_amplifier_serialized.size  # start of the file

        if t_amplifier_serialized.size == 0:
            t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])
        else:
            intan['t_amplifier'] += (t_amplifier_serialized[-1] + (1 / sample_rate['rhd']))
            t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])

        # File information (name, start and end timestamp of each file)
        file_list.append(file.stem)
        file_start_list.append(t_amplifier_serialized[start_ind] * 1E3)  # in ms
        file_end_list.append(t_amplifier_serialized[-1] * 1E3)

        onsets += intan['t_amplifier'][0] * 1E3  # convert to ms
        offsets += intan['t_amplifier'][0] * 1E3

        # Demarcate song bouts
        onset_list.append(demarcate_bout(onsets, intervals))
        offset_list.append(demarcate_bout(offsets, intervals))
        syllable_list.append(demarcate_bout(syllables, intervals))
        context_list.append(context)

    # Organize event-related info into a single dictionary object
    event_info = {
        'file': file_list,
        'file_start': file_start_list,
        'file_end': file_end_list,
        'onsets': onset_list,
        'offsets': offset_list,
        'syllables': syllable_list,
        'context': context_list
    }
    return event_info


class ClusterInfo:

    def __init__(self, database):
        self.database = database  # sqlite3.Row object
        self.id = database['id']
        self.birdID = database['birdID']
        self.taskName = database['taskName']
        self.taskSession = database['taskSession']
        self.taskSessionDeafening = database['taskSessionDeafening']
        self.taskSessionPostDeafening = database['taskSessionPostDeafening']
        self.dph = database['dph']
        self.block10days = database['block10days']
        self.sessionDate = database['sessionDate']
        self.site = database['site']
        self.channel = database['channel']
        self.unit = database['unit']
        self.clusterQuality = database['clusterQuality']
        self.region = database['region']
        self.songNote = database['songNote']
        self.motif = database['motif']
        self.introNotes = database['introNotes']
        self.calls = database['calls']
        self.callSeqeunce = database['callSeqeunce']
        self.format = database['format']  # ephys file format ('rdh', or 'cbin')

        # Get cluster name
        cluster_id = ''
        if len(str(self.id)) == 1:
            cluster_id = '00' + str(self.id)
        elif len(str(self.id)) == 2:
            cluster_id = '0' + str(self.id)
        cluster_taskSession = ''
        if len(str(self.taskSession)) == 1:
            cluster_taskSession = 'D0' + str(str(self.taskSession))
        elif len(str(self.taskSession)) == 2:
            cluster_taskSession = 'D' + str(str(self.taskSession))
        cluster_taskSession += '(' + str(self.sessionDate) + ')'

        cluster_name = [cluster_id, self.birdID, self.taskName, cluster_taskSession, self.sessionDate,
                        self.site, self.channel, self.unit]
        cluster_name = '-'.join(map(str, cluster_name))
        self.name = cluster_name
        print('')
        print('Load cluster {self.name}'.format(self=self))
        # Get cluster path
        project_path = project()
        cluster_path = project_path / self.birdID / self.taskName / cluster_taskSession / self.site[-2:] / 'Songs'
        cluster_path = Path(cluster_path)
        self.path = cluster_path


    def __del__(self):
        print('Delete cluster : {self.name}'.format(self=self))

    def __repr__(self):
        '''Print out the name'''
        return '{self.name}'.format(self=self)


    def load_spk(self, unit='ms'):

        spk_txt_file = list(self.path.glob('*' + self.channel + '(merged).txt'))[0]
        spk_info = np.loadtxt(spk_txt_file, delimiter='\t', skiprows=1)  # skip header
        unit_nb = int(self.unit[-2:])

        # Select only the unit (there could be multiple isolated units in the same file)
        if unit_nb:  # if the unit number is specified
            spk_info = spk_info[spk_info[:, 1] == unit_nb, :]

        spk_ts = spk_info[:, 2]  # spike time stamps
        spk_wf = spk_info[:, 3:]  # spike waveform
        nb_spk = spk_wf.shape[0]  # total number of spikes

        # Units are in second by default, but convert to  millisecond with the argument
        if unit is 'ms':
            spk_ts *= 1E3

        self.spk_ts = spk_ts  # spike timestamps in ms
        self.spk_wf = spk_wf  # individual waveforms
        self.nb_spk = nb_spk  # the number of spikes
        print("spk_ts, spk_wf, nb_spk added")


    def waveform_analysis(self):

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

    def load_raw_data(self, concat=False):

        # List .rhd files
        rhd_files = list(self.path.glob('*.rhd'))

        # Initialize variables
        t_amplifier_serialized = np.array([], dtype=np.float64)
        amplifier_data_serialized = np.array([], dtype=np.float64)

        # Loop through Intan .rhd files
        for file in rhd_files:

            # Load the .rhd file
            intan = read_rhd(file)  # note that the timestamp is in second

            # Serialize time stamps
            intan['t_amplifier'] -= intan['t_amplifier'][0]  # start from t = 0

            if t_amplifier_serialized.size == 0:
                t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])
            else:
                intan['t_amplifier'] += (t_amplifier_serialized[-1] + (1 / sample_rate[self.format]))
                t_amplifier_serialized = np.append(t_amplifier_serialized, intan['t_amplifier'])


    def load_events(self):
        """
        Obtain event info & serialized timestamps for song & neural analysis
        """
        file_name = self.path / 'events.npy'

        if file_name.exists():
            event_dic = np.load(file_name, allow_pickle=True).item()

        else:
            event_dic = get_event_info(self.path)

            # Save event_dic as a numpy object
            np.save(file_name, event_dic)

        # Set the dictionary values to class attributes
        for key in event_dic:
            setattr(self, key, event_dic[key])

        print("files, file_start, file_end, onsets, offsets, syllables, context added")


    def list_files(self):
        files = [file.stem for file in self.path.rglob('*.wav')]
        return files

    def get_nb_files(self):

        nb_files = {}

        nb_files['D'] = len([context for context in self.context if context == 'D'])
        nb_files['U'] = len([context for context in self.context if context == 'U'])
        nb_files['ALL'] = len([context for context in self.context if context == 'U'])
        return nb_files

    def get_nb_bouts(self):
        pass

    def get_nb_motifs(self):
        pass

    def get_peth(self):
        """Get peri-event time histograms & rasters"""
        pass


class RawData(ClusterInfo):
    def __init__(self, database):
        super().__init__(database)

    def load_audio(self):
        pass

    def load_neural_trace(self, channel):
        pass

    def load_timestamp(self):
        pass



