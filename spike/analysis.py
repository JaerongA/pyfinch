from song.functions import *
from spike.parameters import *



def get_event_info(cell_path):
    """
    Obtain serialized event info for song & neural analysis
    """
    import numpy as np
    from spike.load import read_rhd

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
            intan['t_amplifier'] += (t_amplifier_serialized[-1] + (1 / sample_rate['intan']))
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




class SongAnalyzer():
    def __init__(self, spk_ts):
        self.spk_ts = spk_ts
    pass










