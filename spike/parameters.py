"""
By Jaerong
Key parameters for analyzing neural data
"""

sample_rate = 30000  # Intan board (30kHz)
# Define baseline period 1 sec window & 2 sec prior to syllable onset
baseline = {'time_win': 1000, 'time_buffer': 2000}  # in ms