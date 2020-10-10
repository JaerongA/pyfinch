"""
By Jaerong
In some sessions, the raw data were mistakenly loaded as ADBit values in Offline sorter.
It significantly decreased the amplitude of the waveform of those clusters isolated under that setting.
This program converts the amplitude of those clusters by using ADbit value
"""

from database import load
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spike.load import read_spk_txt


def convert_adbit_volts(spk_waveform):
    """Input the waveform matrix extracted from the cluster .txt output"""

    '''Parameters on the Offline Sorter'''
    volt_range = 10  # in milivolts +- 5mV
    sampling_bits = 16
    volt_resolution = 2**sampling_bits

    spk_waveform_new = spk_waveform / ((volt_range / volt_resolution) * 1E3)
    return spk_waveform_new


query = "SELECT * FROM cluster WHERE adbit_cluster IS TRUE"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():
    cell_name, cell_path = load.cell_info(row)
    print('Loading... ' + cell_name)
    unit_nb = int(row['unit'][-2:])

    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    '''Read from the cluster .txt file'''
    spk_ts, spk_waveform, nb_spk = read_spk_txt(spk_txt_file)
    if row['adbit_cluster']:
        print('a')
    spk_waveform_new = convert_adbit_volts(spk_waveform)
    break
    new_spk_txt_file = \
        spk_txt_file.rename(Path(spk_txt_file.parent, f"{spk_txt_file.stem}_new{spk_txt_file.suffix}"))
