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
#
# # Get pre-deafening baseline
# query = f"""SELECT syl.birdID, syl.taskName, song.taskSessionPostDeafening, syl.note, syl.entropy, syl.entropyVar AS ev
#             FROM individual_syllable syl INNER JOIN song
#             ON song.id = syl.songID WHERE syl.context='U' AND syl.taskName='Predeafening'"""
# df_pre = db.to_dataframe(query)
#
# # Get post-deafening where neural data are present
# query = \
#     f"""SELECT birdID, taskName, taskSessionPostDeafening, note, nbNoteUndir AS nbNotes, entropyUndir AS entropy,  entropyVarUndir AS ev
#         FROM syllable_pcc
#         WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit}
#         AND taskName='Postdeafening'
#         """
# df_pcc = db.to_dataframe(query)

# Plot results
def plot_song_feature_pre_post(song_feature,
                               x_lim=None, y_lim=None,
                               fig_name='Untitled', fig_ext='.png', save_fig=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from util.draw import remove_right_top

    bird_list = df_pre['birdID'].unique()

    df_pre = df_pre.groupby(['birdID', 'note']).mean().reset_index()  # Get average feature values per bird per note
    df_pre['nb_notes'] = df_pre.groupby(['birdID', 'note'])[
        'note'].count().values  # Add a column with the number of each note
    df_pre.rename(columns={'nb_notes': 'nbNotePre',
                                'entropy': 'entropyPre',
                                'spectroTemporalEntropy': 'spectroTemporalEntropyPre',
                                'entropyVar': 'entropyVarPre',
                                }, inplace=True)
    df_pre.drop('taskSessionPostDeafening', axis=1, inplace=True)

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

        df_bird_pre = df_pre[df_pre['birdID'] == bird]
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

        # Plot connecting lines
        for note in df_merge['note']:
            ax.plot([0, df_merge['taskSessionPostDeafening'].unique()],
                    [df_merge[df_merge['note'] == note][song_feature + 'Pre'].values,
                     df_merge[df_merge['note'] == note][song_feature + 'Post'].values],
                    'k-', linewidth=1)

    remove_right_top(ax)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel('Days after deafening')
    ax.set_ylabel(song_feature)
    plt.suptitle(f"""Predefeaning mean vs. Last day of neural recording after deafening
                    \n # of notes = {nb_notes}""", y=0.9, fontsize=8)

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
    else:
        plt.show()



# plot_song_feature_pre_post(song_feature='entropy',
#                            # x_lim=[0, 70],
#                            y_lim=[0.3, 1],
#                            fig_name='song_pre_post_entropy', fig_ext='.png', save_fig=False)
#
# plot_song_feature_pre_post(song_feature='entropyVar',
#                            # x_lim=[0, 70],
#                            y_lim=[0, 0.03],
#                            fig_name='song_pre_post_ev', fig_ext='.png', save_fig=False)


# Subsample from either pre- or post-deafening to match the number of rendition
# Decide how many renditions to subsample from predeafening

# Pre
# ['id', 'noteIndSession', 'noteIndFile', 'songID', 'fileID', 'birdID',
#        'taskName', 'note', 'context', 'entropy', 'spectroTemporalEntropy',
#        'entropyVar', 'taskSessionPostDeafening']

# Post
# ['birdID', 'taskName', 'taskSessionPostDeafening', 'note', 'nbNoteUndir',
#        'entropyUndir', 'spectroTempEntropyUndir', 'entropyVarUndir']

song_feature = 'entropy'

def compare_song_feature_pre_post(db, song_feature):

    # Get song features from pre & post (Undir)
    query = f"""SELECT syl.birdID, syl.taskName, song.taskSessionPostDeafening, syl.note, syl.entropy, syl.entropyVar AS ev
                FROM individual_syllable syl INNER JOIN song 
                ON song.id = syl.songID WHERE syl.context='U'
            """
    df = db.to_dataframe(query)

    # Get post-deafening where neural data are present
    query = \
        f"""SELECT birdID, taskName, taskSessionPostDeafening, note, nbNoteUndir AS nbNotes, entropyUndir AS entropy,  entropyVarUndir AS ev 
            FROM syllable_pcc 
            WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit} 
            AND taskName='Postdeafening'
            """
    df_pcc = db.to_dataframe(query)
    df_last_day = df_pcc.groupby(['birdID', 'note'])['taskSessionPostDeafening'].max().reset_index()

    df_results = pd.DataFrame() # store results here

    for _, row in df_last_day.iterrows():
        # if not row['birdID'] == 'b4r64': continue
        bird, note, last_day = row['birdID'], row['note'], row['taskSessionPostDeafening']
        df_post = df.loc[(df['birdID']==bird) &
                         (df['taskSessionPostDeafening'] == last_day) &
                         (df['note'] == note)
                         ]

        nb_notes_post = len(df_post)
        if nb_notes_post < nb_note_crit: continue

        df_pre = df.loc[(df['birdID'] == bird) &
                        (df['taskName'] == 'Predeafening') &
                        (df['note'] == note)
                        ]

        nb_notes_pre = len(df_pre)
        if nb_notes_pre < nb_note_crit: continue
        # Subsample so that # of notes are equal between pre and post
        if nb_notes_pre < nb_notes_post:
            pre_data, post_data = \
                df_pre[song_feature].values, \
                df_post.loc[df_post['note']==note].sample(n=nb_notes_pre, random_state=1)[song_feature].values
        else:
            pre_data, post_data = \
                df_pre.sample(n=nb_notes_post, random_state=1)[song_feature].values, \
                df_post.loc[df_post['note'] == note][song_feature].values
        # print(len(pre_data), len(post_data))
        # print(f"{pre_data.mean() :.3f}, {post_data.mean() :.3f}")

        # rank-sum test (two-sample non-parametric test)
        from scipy.stats import ranksums
        if song_feature == 'entropy':
            stat, pval = ranksums(pre_data, post_data, alternative='less')
        elif  song_feature == 'ev':
            stat, pval = ranksums(pre_data, post_data, alternative='greater')

        # degree_of_freedom = len(pre_data) + len(pre_data) - 2
        # if pval < 0.001:  # mark significance
        #     msg = f"ranksum Z={stat : .3f}, p < 0.001"
        # else:
        #     msg = f"ranksum Z={stat : .3f}, p={pval : .3f}"

        # Organize results
        df_temp = pd.DataFrame({'birdID' : [bird],
                                'note': [note],
                                'nbNotes' : [len(pre_data)],
                                song_feature + 'Pre': [round(pre_data.mean(), 3)],
                                song_feature + 'Post': [round(post_data.mean(), 3)],
                                'postSessionDeafening': [last_day],
                                'sig': [pval<0.05],
                                })
        df_results = df_results.append(df_temp, ignore_index=True)

    return df_results


def plot_song_feature_pre_post_stats(df, song_feature, x_lim=None, y_lim=None):

    # plot_song_feature_pre_post_stats
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from util.draw import remove_right_top

    # Loop through birds to plot all notes
    dot_size = 60
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Plot scatter
    sns.scatterplot(np.zeros(len(df)), df[song_feature + 'Pre'],
                    s=dot_size, color='k', ax=ax)

    sns.scatterplot(np.ones(len(df)) * df['postSessionDeafening'],
                    df[song_feature + 'Post'],
                    s=dot_size, color='k', ax=ax)


    for _, row in df.iterrows():
        color = 'm-' if row['sig'] else 'k-'  # mark significance in magenta
        ax.plot([0, row['postSessionDeafening']],
                [row[song_feature + 'Pre'], row[song_feature + 'Post']],
                color, linewidth=1)

    remove_right_top(ax)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel('Days after deafening')
    ax.set_ylabel(song_feature)
    plt.suptitle(f"""Pre vs. Last day of neural recording after deafening \n
                    \n # of notes = {len(df)} ({df['sig'].sum()} /{len(df_results)})""", y=1, fontsize=10)
    plt.show()

# Entropy
# df_results = compare_song_feature_pre_post(db, song_feature='entropy')
# plot_song_feature_pre_post_stats(df_results, song_feature='entropy',
#                                  x_lim=[-5, 70], y_lim=[0.3, 0.9]
#                                  )
# # Entropy Variance
df_results = compare_song_feature_pre_post(db, song_feature='ev')
# plot_song_feature_pre_post_stats(df_results, song_feature='ev',
#                                  # x_lim=[-5, 70], y_lim=[0.3, 0.9]
#                                  )

# Get post-deafening where neural data are present
query = \
    f"""SELECT birdID, taskName, taskSessionPostDeafening AS postSessionDeafening, note, nbNoteUndir AS nbNotes, pccUndir
        FROM syllable_pcc
        WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit}
        AND taskName='Postdeafening'
        """
df_pcc = db.to_dataframe(query)

df_merged = pd.merge(df_results, df_pcc, how='left', on=['birdID', 'note', 'postSessionDeafening'])

# df_merged.to_csv(r"C:\Users\jahn02\Box\Data\Deafening Project\Analysis\Database\merged_table.csv", header=True)

import matplotlib.pyplot as plt
from deafening.plot import plot_bar_comparison

fig, ax = plt.subplots(figsize=(4, 3))
plot_bar_comparison(ax, df_merged['pccUndir'], df_merged['sig'],
                    hue_var=df_merged['birdID'],
                    title='PCC Undir Post-deafening comparison (EV)',
                    ylabel='PCC',
                    # y_lim=[-0.01, 0.15],
                    xtick_label= ['NonSig', 'Sig'],
                    legend_ok=True
                    )
plt.tight_layout()
plt.show()