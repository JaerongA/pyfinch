"""
Store key parameters for data analysis.

Only the parameters that are used routinely are listed here, but can be changed.

Please note that the changed parameter affect values in other programs.

If one were to play with a range of different parameters, take it out and use it as a local variable in a script.
"""

from typing import Dict, List

import numpy as np

# Song parameters
sample_rate: Dict[str, int] = {
    "rhd": 30000,
    "cbin": 32000,
    "recorder": 44000,
}  # sampling rate for audio signal (Hz)
bout_crit: int = 500  # the temporal criterion for separating song bouts (in ms)
freq_range: List = [
    300,
    8000,
]  # frequency range for bandpass filter for spectrogram (Hz)

# Set syllable colors
note_color: Dict[str, List[str]] = {
    "Intro": ["k", "gray", "darkseagreen", "olive"],
    "Motif": ["r", "b", "lime", "m", "brown", "purple", "saddlebrown"],
    "Call": [
        "teal",
        "darkgrey",
        "indigo",
        "darkgray",
    ],
}

bout_color: Dict[str, str] = {
    "i": "k",
    "j": "gray",
    "k": "darkseagreen",  # intro notes
    "a": "r",
    "b": "b",
    "c": "lime",
    "d": "m",
    "e": "brown",
    "f": "purple",
    "g": "saddlebrown",  # motif notes
    "m": "teal",
    "n": "darkgrey",
    "l": "darkgray",
    "o": "indigo",  # calls
    "A": "w",
}

# Define baseline period 1 sec window & 2 sec prior to syllable onset
baseline: Dict[str, int] = {"time_win": 1000, "time_buffer": 2000}  # in ms

# For analysis correlogram
spk_corr_parm = {"bin_size": 1, "lag": 100}  # in ms

if spk_corr_parm["lag"] % spk_corr_parm["bin_size"]:
    raise Exception(
        "lag should be divisible by bin size (e.g., bin_size = 2, lag = 100"
    )

spk_corr_parm["time_bin"] = np.arange(
    -spk_corr_parm["lag"],
    spk_corr_parm["lag"] + spk_corr_parm["bin_size"],
    spk_corr_parm["bin_size"],
)

# For peth (peri-event time histogram) or rasters
peth_parm = {
    "buffer": 50,  # time buffer before the event onset (in ms)
    "bin_size": 1,  # peth time bin size (in ms)
    "nb_bins": 1500,  # number of bins
}
peth_parm["time_bin"] = np.arange(0, peth_parm["nb_bins"], peth_parm["bin_size"])

# Gauss parameter for PETH smoothing
gauss_std = 1  # experiment with 0.5, 1, 3. Previously used 8
filter_width = 20  # filter length for smoothing (in ms)
# truncate = (((filter_width - 1)/2)-0.5)/ gauss_std

spk_count_parm = {
    "win_size": 30
}  # moving window where number of spikes will be calculated (in ms)

# Raster
tick_length = 0.8
tick_width = 0.7

# Bursting criteria
burst_hz = 200  # instantaneous firing rates >= 200 Hz to be considered bursting
corr_burst_crit = 5  # bursting criteria (in ms) in a correlogram, peak latency of a correlogram should be <= than the criteria

# ISI analysis
isi_win = 4  # 10^isi_win ms (log scale)
isi_scale = 100
isi_bin = np.arange(0, isi_win, 1 / isi_scale)

# Correlogram
corr_shuffle = {
    "shuffle_limit": 5,  # in ms
    "shuffle_iter": 100,  # bootstrap iterations
}

# shuffling_iter = 100  # shuffling iteration for obtaining baseline

# Add a random spike jitter
jitter_limit = 5  # maximum amount of jitter (in ms)

# Bout raster plot
bout_buffer = 500  # in ms

# Note temporal buffer (for single syllable)
note_buffer = 10  # in ms

# Pre-motor window spike calculation
pre_motor_win_size = 50  # in ms
post_song_win_size = 50  # in ms

nb_note_crit = 10  # minimum number of notes for analysis
fr_crit = 10  # minimum firing rates criterion (in Hz)

# Spike shuffling parameter for peth for getting baseline PCC
peth_shuffle = {
    "shuffle_limit": 10,  # in ms (50 for motif, 10 for syllable)
    "shuffle_iter": 100,  # bootstrap iterations
}

alpha = 0.05  # for stat significance

# Sequence analysis
cmap_list = [
    "YlGnBu",
    "PiYG",
]
sequence_color = {
    "song_note": ["r", "b", "lime", "m", "darkorange", "purple", "saddlebrown"],
    "intro": ["k", "gray", "darkseagreen", "olive"],
    "call": ["teal", "darkslategray", "darkgray", "indigo"],
}

# Waveform analysis
interp_factor = 100  # factor by which to increase the sampling frequency
spk_proportion = 0.2  # proportion of waveforms to plot (0 to 1 (all waveforms))
