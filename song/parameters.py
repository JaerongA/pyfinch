"""
By Jaerong
Key parameters for analyzing behavior (Song)
"""

bout_crit = 500  # the temporal criterion for separating song bouts (in ms)
sample_rate = {'intan': 30000, 'recorder': 44000}  # sampling rate for audio signal (Hz)
freq_range = [300, 8000]  # frequency range for bandpass filter for spectrogram (Hz)


def get_syl_color():
    # color for each syllable
    import numpy as np
    syl_color = np.zeros((5, 3))  # colors for song notes
    syl_color[0, :] = [247, 198, 199]
    syl_color[1, :] = [186, 229, 247]
    syl_color[2, :] = [187, 255, 195]
    syl_color[3, :] = [249, 170, 249]
    syl_color[4, :] = [255, 127.5, 0]
    syl_color = syl_color / 255
    return syl_color
