"""
By Jaerong
Key parameters for analyzing neural data
"""

sample_rate = {'rhd': 30000, 'cbin': 32000, 'recorder': 44000}  # sampling rate for audio signal (Hz)

# Define baseline period 1 sec window & 2 sec prior to syllable onset
baseline = {'time_win': 1000, 'time_buffer': 2000}  # in ms

# For spike correlogram
corr = {'bin_size': 1, 'lag': 100}  # in ms

# For peth (peri-event time histogram)
peth = {'buffer': 500  # time buffer before the event onset in ms
        }
