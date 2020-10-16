"""
By Jaerong
A collection of functions used for song analysis
"""


def syl_type_(syllable, cluster):
    """ function to determine the category of the syllable """
    type_str = []
    for syllable in syllable:
        if syllable in cluster.Motif:
            type_str.append('M')  # motif
        elif syllable in cluster.Calls:
            type_str.append('C')  # call
        elif syllable in cluster.IntroNotes:
            type_str.append('I')  # intro notes
        else:
            type_str.append(None)  # intro notes
    return type_str


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
