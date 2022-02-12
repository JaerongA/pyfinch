"""
Convert audio files with no extension (Mimi's data) into .wav format
e.g., undir.20120128.1850 -> k27o36_120128_1850_Undir.wav
undir.20120128.1855.not
"""

##TODO: this file contains a bug, needs to be fixed before use (.not.mat name doesn't match) (2021/04/11 Jaerong)
from pathlib import Path
from ..utils.functions import find_data_path

data_path = Path(r"C:\Users\jahn02\Box\Data\Deafening Project\o25w75\Predeafening\D01(20120208)\01\Undir")
bird_id = 'o25w75'
context = 'Undir'

if data_path is None:
    data_path = find_data_path()

audio_files = [file for file in data_path.rglob('*') if file.suffix != '.mat']  # files to read & convert
not_mat_files = [file for file in data_path.rglob('*') if file.suffix == '.mat']

for audio_file, not_mat_file in zip(audio_files, not_mat_files):
    date = audio_file.name.split('.')[1][2:]
    file_ind = audio_file.name.split('.')[-1]

    new_audio_name = '_'.join([bird_id, date, file_ind, context])  # e.g., k27o36_120128_1850_Undir.wav
    new_audio_name += '.wav'
    new_not_mat_name = new_audio_name + '.not.mat'

    new_audio_name = audio_file.parent / new_audio_name
    new_not_mat_name = not_mat_file.parent / new_not_mat_name
    print(f"Changing '{audio_file.name}' into '{new_audio_name}'")
    audio_file.rename(new_audio_name)
    not_mat_file.rename(new_not_mat_name)

print('Done!')
