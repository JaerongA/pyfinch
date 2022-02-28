"""
Plot note stats (distribution of syllable duration)
"""


def plot_notestat(data_path=None, save_path=None, save_fig=False, ext='.png'):
    """
    Plot syllable durations to view distributions & detect outliers
    This should done after each syllable segmentation with uisonganal.m
    Labeling files should exist in .not.mat format in the data path

    Parameters
    ----------
    data_path : path
        The folder that contains labeling .not.mat files
    save_fig : bool
        If true, save result figure. If False, just display it

    """

    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    import seaborn as sns
    from ..analysis.load import read_not_mat
    from ..utils.draw import remove_right_top
    from ..utils.functions import myround, open_folder, find_data_path
    from ..utils import save
    import math

    # Find data path

    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    # Store results in the dataframe
    audio_files = list(data_path.glob('*.wav'))

    df = pd.DataFrame()

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
                  palette=sns.color_palette(),  # set color category to be consistent regardless of the number of syllables
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

    if save_fig:
        save_fig(fig, data_path, data_path.name, ext=ext)
    else:
        plt.show()
    open_folder(data_path)
    print("Done!")


    # # Save results
    # if save_fig:
    #     save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name)
    #     save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext, view_folder=True)
    # else:
    #     plt.show()

data_dir = r"H:\Box\Project\sample_data\Songs"
plot_notestat(data_path=data_dir)