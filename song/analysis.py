"""
By Jaerong
A collection of functions used for song analysis
"""

def read_not_mat(notmat, unit='ms'):
    """ read from .not.mat files generated from uisonganal
    Parameters
    ----------
    notmat : str
        Name of the .not.mat file (path)
    unit : (optional)
        milli-second by default. Convert to seconds when specified

    Returns
    -------
    onsets : array
        time stamp for syllable onset (in ms)
    offsets : array
        time stamp for syllable offset (in ms)
    intervals : array
        temporal interval between syllables (i.e. syllable gaps) (in ms)
    duration : array
        duration of each syllable (in ms)
    syllables : str
        song syllables
    context : str
        social context ('U' for undirected and 'D' for directed)
    """
    import scipy.io

    onsets = scipy.io.loadmat(notmat)['onsets'].transpose()[0]  # syllable onset timestamp
    offsets = scipy.io.loadmat(notmat)['offsets'].transpose()[0]  # syllable offset timestamp
    intervals = onsets[1:] - offsets[:-1]  # syllable gap durations (interval)
    duration = offsets - onsets  # duration of each syllable
    syllables = scipy.io.loadmat(notmat)['syllables'][0]  # Load the syllable info
    context = notmat.name.split('.')[0].split('_')[-1][0].upper()  # extract 'U' (undirected) or 'D' (directed) from the file name
    if context is not 'U' or context is not 'D':  # if the file was not tagged with Undir or Dir
        context = None


    # units are in ms by default, but convert to second with the argument
    if unit is 'second':
        onsets /= 1E3
        offsets /= 1E3
        intervals /= 1E3
        duration /= 1E3

    return onsets, offsets, intervals, duration, syllables, context


def syl_type_(syllables, song_info):
    """ function to determine the category of the syllable
    INPUT1: syllables (str)
    INPUT2: intervals (array) syllable gap duration
    OUTPUT: type of the syllable (e.g., motif, intro notes, calls)
    """
    syl_type = []
    for syllable in syllables:
        if syllable in song_info['motif']:
            syl_type.append('M')  # motif
        elif syllable in song_info['calls']:
            syl_type.append('C')  # call
        elif syllable in song_info['introNotes']:
            syl_type.append('I')  # intro notes
        else:
            syl_type.append(None)
    return syl_type


def demarcate_bout(target, intervals):
    """ Demarcate the song bout with an asterisk (*) from a string of syllables
    Parameters
    ----------
    target : str or numpy array
    intervals_ms : int
        syllable gap duration in ms

    Returns
    -------
    bout_labeling : str
        demarcated syllable string (e.g., 'iiiabc*abckn*')
    """
    from song.parameters import bout_crit
    import numpy as np

    ind = np.where(intervals > bout_crit)[0]
    bout_labeling = target

    if isinstance(target, str):
        if len(ind):
            for i, item in enumerate(ind):
                if i is 0:
                    bout_labeling = target[:item + 1]
                else:
                    bout_labeling += '*' + target[ind[i - 1] + 1:ind[i] + 1]
            bout_labeling += '*' + target[ind[i] + 1:]

        bout_labeling += '*'  # end with an asterisk


    elif isinstance(target, np.ndarray):
        if len(ind):
            for i, item in enumerate(ind):
                if i is 0:
                    bout_labeling = target[:item + 1]
                else:
                    bout_labeling = np.append(bout_labeling, '*')
                    bout_labeling = np.append(bout_labeling, target[ind[i - 1] + 1: ind[i] + 1])

            bout_labeling = np.append(bout_labeling, '*')
            bout_labeling = np.append(bout_labeling, target[ind[i] + 1:])

        bout_labeling = np.append(bout_labeling, '*')  # end with an asterisk

    return bout_labeling


def unique_nb_notes_in_bout(note: str, bout: str):
    """ returns the unique number of notes within a single bout string """
    nb_song_note_in_bout = len([note for note in note if note in bout])
    return nb_song_note_in_bout


def total_nb_notes_in_bout(note: str, bout: str):
    """ returns the total number of song notes from a list of song bouts"""
    notes = []
    nb_notes = []
    for note in note:
        notes.append(note)
        nb_notes.append(sum([bout.count(note) for bout in bout]))
    return sum(nb_notes)


def get_nb_bouts(song_note: str, bout_labeling: str):
    """ Count the number of bouts (only includes those having at least one song note)
    INPUT1: song_note (e.g., abcd, syllables that are part of a motif)
    INPUT2: bout_labeling (e.g., iiiiiiiiabcdjiiiabcdji*, syllables that are demarcated by * (bout))
    """
    nb_bouts = len([bout for bout in bout_labeling.split('*')[:-1] if
                    unique_nb_notes_in_bout(song_note, bout)])
    return nb_bouts


class SongInfo:

    def __init__(self, database):
        self.database = database
        pass
    pass
