"""
Utility functions for drawing spectrograms

Spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion

by David Nicholson https://github.com/NickleDave/vak
"""


def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram(dat, samp_freq, fft_size=512, step_size=64, thresh=None, transform_type=None,
                freq_range=None):
    """creates a spectrogram

    Parameters
    ----------
    dat : numpy.ndarray
        audio signal
    samp_freq : int
        sampling frequency in Hz
    fft_size : int
        size of window for Fast Fourier transform, number of time bins.
    step_size : int
        step size for Fast Fourier transform
    transform_type : str
        one of {'log_spect', 'log_spect_plus_one'}.
        'log_spect' transforms the spectrogram to log(spectrogram), and
        'log_spect_plus_one' does the same thing but adds one to each element.
        Default is None. If None, no transform is applied.
    thresh: int
        threshold minimum power for log spectrogram
    freq_range : tuple
        of two elements, lower and higher frequencies.

    Return
    ------
    spect : numpy.ndarray
        spectrogram
    freqbins : numpy.ndarray
        vector of centers of frequency bins from spectrogram

    """
    import numpy as np
    from matplotlib.mlab import specgram

    noverlap = fft_size - step_size

    if freq_range:
        dat = butter_bandpass_filter(dat,
                                     freq_range[0],
                                     freq_range[1],
                                     samp_freq)

    # below only take [:3] from return of specgram because we don't need the image
    spect, freqbins, timebins = specgram(dat, fft_size, samp_freq, noverlap=noverlap)[:3]

    if transform_type:
        if transform_type == 'log_spect':
            spect /= spect.max()  # volume normalize to max 1
            spect = np.log10(spect)  # take log
            if thresh:
                # I know this is weird, maintaining 'legacy' behavior
                spect[spect < -thresh] = -thresh
        elif transform_type == 'log_spect_plus_one':
            spect = np.log10(spect + 1)
            if thresh:
                spect[spect < thresh] = thresh
    else:
        if thresh:
            spect[spect < thresh] = thresh  # set anything less than the threshold as the threshold

    if freq_range:
        f_inds = np.nonzero((freqbins >= freq_range[0]) &
                            (freqbins < freq_range[1]))[0]  # returns tuple
        spect = spect[f_inds, :]
        freqbins = freqbins[f_inds]

    return spect, freqbins, timebins
