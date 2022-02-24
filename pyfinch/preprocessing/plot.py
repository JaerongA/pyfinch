"""
Plot raw data before further preprocessing (e.g., raw data, syllable durations, etc)
"""


def notestat(data_path=None, fig_name=None, save_path=None, save_fig=True,
             fig_ext='.png', view_folder=True):
    """
    Plot syllable durations to view distributions & detect outliers.

    This should done after each syllable segmentation with uisonganal.m

    Labeling files should exist in .not.mat format in the data path.

    Parameters
    ----------
    data_path : path
        The folder that contains labeling .not.mat files.
    fig_name : str, optional
        Name of the figure. If not specified, defaults to the data path name.
    save_path : path, optional
        Path to save the figure. If not specified, save to the data path.
    save_fig : bool, default=True
        If true, save result figure. If False, just display it.
    fig_ext : str, default='.png'
        Figure file extension
    view_folder : bool, default=True
        Open the save folder
    """

    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from ..analysis.load import read_not_mat
    from ..utils import save
    from ..utils.draw import remove_right_top
    from ..utils.functions import myround, open_folder, find_data_path

    # Find data path
    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    # Store results in the dataframe
    audio_files = list(data_path.glob('*.wav'))

    df = pd.DataFrame()

    # Loop over all the audio files
    for file in audio_files:
        # Load the .not.mat file
        print('Loading... ' + file.stem)
        notmat_file = file.with_suffix('.wav.not.mat')

        if not notmat_file.exists():
            raise FileNotFoundError
            # raise("File doesn't exist!")

        birdID = file.name.split('_')[0]
        onsets, offsets, intervals, durations, syllables, context = read_not_mat(notmat_file)

        nb_syllable = len(syllables)

        temp_df = pd.DataFrame({'FileID': [notmat_file] * nb_syllable,
                                'Syllable': list(syllables),
                                'Duration': durations,
                                })
        df = df.append(temp_df, ignore_index=True)

    # Plot the results
    syllable_list = sorted(list(set(df['Syllable'].to_list())))

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.title("{} - {}".format(birdID, data_path.name))
    sns.stripplot(ax=ax, data=df, x='Syllable', y='Duration',
                  order=syllable_list,
                  s=4,
                  palette=sns.color_palette(),
                  # set color category to be consistent regardless of the number of syllables
                  jitter=0.15)

    for syllable, x_loc in zip(syllable_list, ax.get_xticks()):
        nb_syllable = df[df['Syllable'] == syllable]['Syllable'].count()
        max_dur = df[df['Syllable'] == syllable]['Duration'].max()
        text = "({})".format(nb_syllable)
        x_loc -= ax.get_xticks()[-1] * 0.03
        y_loc = max_dur + ax.get_ylim()[1] * 0.05
        plt.text(x_loc, y_loc, text)

    ax.set_ylim([0, myround(math.ceil(ax.get_ylim()[1]), base=50)])
    remove_right_top(ax)
    plt.ylabel('Duration (ms)')
    fig.tight_layout()

    open_folder(data_path)
    print("Done!")

    # Save the figure
    if save_fig:
        if not save_path:
            save_path = data_path
        if not fig_name:
            fig_name = data_path.name
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder)
    else:
        plt.show()
