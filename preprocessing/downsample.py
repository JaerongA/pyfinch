"""Downsample .wav files """


import librosa
from pathlib import Path
import soundfile as sf
from util.functions import find_data_path

target_sr = 32000  # target sampling rate

# Specify dir here or search for the dir manually
data_dir = Path(r'H:\Box\Data\Deafening Project\o25w75\Predeafening\D01(20120208)\01\Songs')
try:
    data_dir
except NotADirectoryError:
    data_dir = find_data_path()

files = list(data_dir.glob('*.wav'))

for file in files:
    print('Processing... ' + file.stem)
    signal, _ = librosa.load(file, sr=target_sr) # Downsample to the target sample rate
    # sf.write(file, signal, target_sr)
    sf.write(data_dir / file, signal, target_sr)