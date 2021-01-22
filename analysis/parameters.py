"""
By Jaerong
Key parameters for analyzing neural data
"""
import numpy as np

# Song parameters
sample_rate = {'rhd': 30000, 'cbin': 32000, 'recorder': 44000}  # sampling rate for audio signal (Hz)
bout_crit = 500  # the temporal criterion for separating song bouts (in ms)
freq_range = [300, 8000]  # frequency range for bandpass filter for spectrogram (Hz)

# Set syllable colors
note_color = {'Intro': ['k', 'gray', 'darkseagreen', 'olive'],
              'Motif': ['r', 'b', 'lime', 'm', 'brown', 'purple', 'saddlebrown'],
              'Call': ['teal', 'darklategray', 'darkgray', 'indigo']}

# Define baseline period 1 sec window & 2 sec prior to syllable onset
baseline = {'time_win': 1000, 'time_buffer': 2000}  # in ms

# For analysis correlogram
spk_corr_parm = {'bin_size': 1, 'lag': 100}  # in ms

if spk_corr_parm['lag'] % spk_corr_parm['bin_size']:
    raise Exception("lag should be divisible by bin size (e.g., bin_size = 2, lag = 100")

spk_corr_parm['time_bin'] = np.arange(-spk_corr_parm['lag'], spk_corr_parm['lag'] + spk_corr_parm['bin_size'],
                                      spk_corr_parm['bin_size'])

# For peth (peri-event time histogram) or rasters
peth_parm = {'buffer': 50,  # time buffer before the event onset (in ms)
             'bin_size': 1,  # peth time bin size (in ms)
             'nb_bins': 1500  # number of bins
             }
peth_parm['time_bin'] = np.arange(0, peth_parm['nb_bins'], peth_parm['bin_size'])

gauss_filter = 40  # for peth smoothing (in ms)
gauss_std = 8
# gauss_filter = gausswin(filter_size);  %% Gaussian smoothing parameter
# gauss_filter = gauss_filter/sum(gauss_filter);


# Raster
tick_length = 1
tick_width = 1






# def get_syl_color():
#     # color for each syllable
#     import numpy as np
#     syl_color = np.zeros((5, 3))  # colors for song notes
#     syl_color[0, :] = [247, 198, 199]
#     syl_color[1, :] = [186, 229, 247]
#     syl_color[2, :] = [187, 255, 195]
#     syl_color[3, :] = [249, 170, 249]
#     syl_color[4, :] = [255, 127.5, 0]
#     syl_color = syl_color / 255
#     return syl_color


# note_color = []
#
# for i, note in enumerate(note_seq[:-1]):
#     if note in song_note:
#         note_color.append(motif_color.pop(0))
#     elif note in intro_notes:
#         note_color.append(intro_color.pop(0))
#     elif note in calls:
#         note_color.append(call_color.pop(0))
#     else:
#         note_color.append(intro_color.pop(0))
#
# note_color.append('y')  # for stop at the end
