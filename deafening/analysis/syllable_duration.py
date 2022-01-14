"""
Syllable duration for all syllables regardless of it type
Calculation based on EventInfo.m
"""


def get_duration(query):
    from analysis.functions import get_note_type
    from analysis.song import SongInfo
    from database.load import ProjectLoader, DBInfo
    import pandas as pd
    from util import save

    # Create save path
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'SyllableDuration')

    # Load song database
    db = ProjectLoader().load_db()
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():
        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        # Load song object
        si = SongInfo(path, name)
        print('\nAccessing... ' + si.name)

        # Store results in the dataframe
        df = pd.DataFrame()

        # Organize results
        note_types = get_note_type(''.join(si.syllables).replace('*', ''), song_db)  # Type of the syllables
        nb_notes = len(''.join(si.syllables).replace('*', ''))
        durations = []
        for duration in [list(map(float, duration[duration != '*'])) for duration in si.durations]:
            durations += duration[0]

        # Save results to a dataframe
        temp_df = []
        temp_df = pd.DataFrame({'SongID': [song_db.id] * nb_notes,
                                'BirdID': [song_db.birdID] * nb_notes,
                                'TaskName': [song_db.taskName] * nb_notes,
                                'TaskSession': [song_db.taskSession] * nb_notes,
                                'TaskSessionDeafening': [song_db.taskSessionDeafening] * nb_notes,
                                'TaskSessionPostdeafening': [song_db.taskSessionPostDeafening] * nb_notes,
                                'Context': ''.join([len(syllable.replace('*', '')) * context for syllable, context
                                                    in zip(si.syllables, si.contexts)]),
                                'SyllableType': note_types,
                                'Syllable': list(''.join(si.syllables).replace('*', '')),
                                'Duration': durations,
                                })
        df = df.append(temp_df, ignore_index=True)

        # Save to a file
        df.index.name = 'Index'
        outputfile = save_path / 'SyllableDuration.csv'
        df.to_csv(outputfile, index=True, header=True)  # save the dataframe to .cvs format
        print('Done!')


def load_data(data_file, context='ALL', syl_type='ALL'):
    import pandas as pd

    # Load syllable duration .csv data
    df = pd.read_csv(data_file)

    # Select syllables based on social context
    if context is 'U':
        df = df.query('Context == "U"')  # select only Undir
    elif context is 'D':
        df = df.query('Context == "D"')  # select only Dir

    # Only select syllables of a particular type
    if syl_type is 'M':
        df = df.query('SyllableType == "M"')  # eliminate non-labeled syllables (e.g., 0)
    elif syl_type is 'C':
        df = df.query('SyllableType == "C"')  # eliminate non-labeled syllables (e.g., 0)
    elif syl_type is 'I':
        df = df.query('SyllableType == "I"')  # eliminate non-labeled syllables (e.g., 0)
    return df


if __name__ == '__main__':

    # Check if the data .csv exists
    from database.load import ProjectLoader

    data_file = ProjectLoader().path / 'Analysis' / 'SyllableDuration' / 'SyllableDuration.csv'

    if not data_file.exists():
        # Get the syllable duration data if it already exists
        # Load song database
        query = "SELECT * FROM song WHERE id=1"
        # query = "SELECT * FROM song WHERE id BETWEEN 1 AND 16"
        get_duration(query)
        df = load_data(data_file, context='ALL', syl_type='ALL')
    else:  # Get the syllable duration data if it already exists
        df = load_data(data_file, context='ALL', syl_type='ALL')

    # Plot the results
    import matplotlib.pyplot as plt
    import seaborn as sns
    import IPython
    from util.functions import unique
    from util import save
    from database.load import ProjectLoader

    fig_ext = '.png'

    bird_list = unique(df['BirdID'].tolist())
    task_list = unique(df['TaskName'].tolist())

    for bird in bird_list:

        for task in task_list:

            note_list = unique(df['Syllable'].tolist())

            # bird = 'b70r38'
            # task = 'Predeafening'

            temp_df = []
            temp_df = df.loc[(df['BirdID'] == bird) & (df['TaskName'] == task)]

            if temp_df.empty:
                continue

            note_list = unique(temp_df.query('SyllableType == "M"')['Syllable'])  # only motif syllables

            title = '-'.join([bird, task])
            fig = plt.figure(figsize=(6, 5))
            plt.suptitle(title, size=10)
            # ax = sns.distplot((temp_df['Duration'], hist= False, kde= True)
            ax = sns.kdeplot(temp_df['Duration'], bw=0.05, label='', color='k', linewidth=2)
            # kde = zip(ax.get_lines()[0].get_duration()[0], ax.get_lines()[0].get_duration()[1])
            # sns.rugplot(temp_df['Duration'])  # shows ticks

            # https: // stackoverflow.com / questions / 43565892 / python - seaborn - distplot - y - value - corresponding - to - a - given - x - value
            # TODO: extrapolate value and mark with an arrow

            # mark each note
            median_dur = list(zip(note_list, temp_df.query('SyllableType == "M"').groupby(['Syllable'])[
                'Duration'].mean().to_list()))  # [('a', 236.3033654971783), ('b', 46.64262295081962), ('c', 154.57333333333335), ('d', 114.20039483457349)]
            for note, dur in median_dur:
                plt.axvline(dur, color='k', linestyle='dashed', linewidth=1)
                plt.arrow(dur, 5, 0, -1)
            ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
            plt.xlabel('Duration (ms)')
            plt.ylabel('Probability Density')
            # plt.xlim(0, ceil(max(df.loc[(df['BirdID'] == bird)]['Duration']) / 100) * 100)
            plt.xlim(0, 300)
            plt.ylim(0, 0.06)

            # print('Prcessing... {} from Bird {}'.format(task, bird))

            # Save results
            # save_dir = project_path / 'Analysis' / 'SyllableDuration'

            # Create save path
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'SyllableDuration')
            save.save_fig(fig, save_path, title, fig_ext=fig_ext)
