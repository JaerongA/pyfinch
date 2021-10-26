"""
Compare song features before and after deafening
Post-deafening songs were obtained from the last session where neural data are present
Use individual_syllable and syllable_pcc tables
"""

from analysis.parameters import fr_crit, nb_note_crit
from database.load import ProjectLoader
from util import save
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load database
db = ProjectLoader().load_db()

# Get pre-deafening baseline
query = f"""SELECT syl.*, song.taskSessionPostDeafening
        FROM individual_syllable syl INNER JOIN song 
        ON song.id = syl.songID WHERE syl.context='U' AND syl.taskName='Predeafening'"""

# Get post-deafening where neural data are present
df = db.to_dataframe(query)
query = f"""SELECT birdID, taskName, taskSessionPostDeafening, note, nbNoteUndir, entropyUndir, spectroTempEntropyUndir, entropyVarUndir  
FROM syllable_pcc 
WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit} 
AND taskName='Postdeafening'
"""
df_pcc = db.to_dataframe(query)


# Plot results
def plot_song_feature_pre_post(song_feature,
                               x_lim=None, y_lim=None,
                               fig_name='Untitled', fig_ext='.png', save_fig=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    bird_list = df['birdID'].unique()

    df_baseline = df.groupby(['birdID', 'note']).mean().reset_index()  # Get average feature values per bird per note
    df_baseline['nb_notes'] = df.groupby(['birdID', 'note'])[
        'note'].count().values  # Add a column with the number of each note
    df_baseline.rename(columns={'nb_notes': 'nbNotePre',
                                'entropy': 'entropyPre',
                                'spectroTemporalEntropy': 'spectroTemporalEntropyPre',
                                'entropyVar': 'entropyVarPre',
                                }, inplace=True)
    df_baseline.drop('taskSessionPostDeafening', axis=1, inplace=True)

    # Get post-deafening where neural data are present
    # Store values from the last day of neural recording in post-deafening in df_post
    df_post = pd.DataFrame()

    for bird in bird_list:
        df_bird = df_pcc[df_pcc['birdID'] == bird]
        df_temp = df_bird[df_bird['taskSessionPostDeafening'] == df_bird['taskSessionPostDeafening'].max()]
        total_nb_notes = df_temp.groupby(['birdID', 'note'])[
            'nbNoteUndir'].sum().values  # Add a column with the number of each note
        df_temp = df_temp.groupby(['birdID', 'note']).mean().reset_index()  # get the mean feature value for the session
        df_temp['nbNoteUndir'] = total_nb_notes
        df_post = df_post.append(df_temp, ignore_index=True)

    # Remove the notes if the number of notes is less than the criteria
    df_post.drop(df_post[df_post['nbNoteUndir'] < nb_note_crit].index, inplace=True)
    # Change column names to match pre-deafening
    df_post.rename(columns={'nbNoteUndir': 'nbNotePost',
                            'entropyUndir': 'entropyPost',
                            'spectroTempEntropyUndir': 'spectroTemporalEntropyPost',
                            'entropyVarUndir': 'entropyVarPost',
                            }, inplace=True)

    # Loop through birds to plot all notes
    dot_size = 60
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    nb_notes = 0  # total number of notes plotted (that has both pre and post data)

    for bird in bird_list:

        # if bird == 'y44r34': continue
        # print(bird)

        df_bird_pre = df_baseline[df_baseline['birdID'] == bird]
        df_bird_post = df_post[df_post['birdID'] == bird]

        # Skip if one of the conditions does not exist
        if df_bird_pre.empty or df_bird_post.empty:
            continue

        # Merge pre & post
        df_merge = pd.merge(df_bird_pre, df_bird_post, how='inner', on=['note', 'birdID'])
        df_merge.drop(
            ['id', 'noteIndSession', 'noteIndFile', 'songID'],
            axis=1, inplace=True)
        nb_notes += len(df_merge)

        # Plot scatter
        sns.scatterplot(np.zeros(len(df_merge)), df_merge[song_feature + 'Pre'],
                        s=dot_size, color='k', ax=ax)

        sns.scatterplot(np.ones(len(df_merge)) * df_merge['taskSessionPostDeafening'].unique(),
                        df_merge[song_feature + 'Post'],
                        s=dot_size, color='k', ax=ax)

        # Connect two dots
        for note in df_merge['note']:
            ax.plot([0, df_merge['taskSessionPostDeafening'].unique()],
                    [df_merge[df_merge['note'] == note][song_feature + 'Pre'].values,
                     df_merge[df_merge['note'] == note][song_feature + 'Post'].values],
                    'k-', linewidth=1)

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel('Days after deafening')
    ax.set_ylabel(song_feature)
    plt.suptitle(f"""Pre-defeaning mean vs. Last day of neural recording after deafening
                    \n # of notes = {nb_notes}""", y=0.9, fontsize=8)

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
    else:
        plt.show()


# Song features to compare (select from these features)
features = ['entropy', 'spectroTemporalEntropy', 'entropyVar']

# plot_song_feature_pre_post(song_feature='entropy',
#                            # x_lim=[0, 70],
#                            y_lim=[0.3, 1],
#                            fig_name='song_pre_post_entropy', fig_ext='.png', save_fig=False)
#
# plot_song_feature_pre_post(song_feature='entropyVar',
#                            # x_lim=[0, 70],
#                            y_lim=[0, 0.03],
#                            fig_name='song_pre_post_ev', fig_ext='.png', save_fig=False)


# Subsample from pre-deafening to match the number of rendition to that from post-deafening
# df_baseline = df.groupby(['birdID', 'note']).mean().reset_index()
bird = 'b70r38'