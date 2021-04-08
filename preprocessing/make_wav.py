"""
By Jaerong
Convert audio files with no extension (Mimi's data) into .wav format
e.g., undir.20120128.1850 -> k27o36_120128_1850_Undir.wav
"""

from pathlib import Path
from utilities.functions import find_data_path

data_path = Path(r"H:\Box\Data\Deafening Project\k27o36\Predeafening\D01(20120130)\01\Undir")
bird_id = 'k27o36'
context = 'Undir'

if data_path is None:
    data_path = find_data_path()

files = [file for file in data_path.rglob('*') if file.suffix != '.mat']  # files to read & convert

for file in files:
    print("Processing ..." + file.name)
    date = file.name.split('.')[1][2:]
    file_ind = file.name.split('.')[-1]

    new_file_name = '_'.join([bird_id, date, file_ind])  # e.g., k27o36_120330_188
