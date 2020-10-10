"""
By Jaerong
load neural data (from .rhd or .txt)
"""


def read_spk_txt(spk_txt_file, *unit_nb):
    """
    Read the output .txt from the Offline Sorter
    Column header of the input .txt
    ['Channel', 'Unit', 'Timestamp']
    Disregard the first column since it is always 1
    Column 3 to 35 stores waveforms
    """
    import numpy as np

    spk_info = np.loadtxt(spk_txt_file, delimiter='\t', skiprows=1)  # skip header

    '''select only the unit (there could be multiple isolated units in the same file)'''
    if unit_nb:  # if the unit number is specified
        spk_info = spk_info[spk_info[:, 1] == unit_nb, :]

    spk_ts = spk_info[:, 2]  # spike time stamps
    spk_waveform = spk_info[:, 3:]  # spike waveform
    nb_spk = spk_waveform.shape[0]  # total number of spikes
    return spk_ts, spk_waveform, nb_spk


