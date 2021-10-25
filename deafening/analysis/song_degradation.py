"""
Compare song features before and after deafening
Post-deafening songs were obtained from the last session where neural data are present
"""

from analysis.parameters import fr_crit, nb_note_crit
from database.load import ProjectLoader
from util import save
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = ['entropy', 'spectroTemporalEntropy', 'entropyVar']

# Load database
db = ProjectLoader().load_db()
# SQL statement
# Select Undir & Predeafening
query = f"""SELECT syl.*, song.taskSessionPostDeafening
        FROM individual_syllable syl INNER JOIN song 
        ON song.id = syl.songID WHERE syl.context='U' AND syl.taskName='Predeafening'"""
df = db.to_dataframe(query)

bird_id = 'w16w14'
bird_list = df['birdID'].unique()

#df_bird = df[df['birdID'] == bird_id]

# Get pre-deafening baseline
df_baseline = df.groupby(['birdID','note']).mean().reset_index()  # Get average feature values per bird per note
df_baseline['nb_notes'] = df.groupby(['birdID','note'])['note'].count().values  # Add a column with the number of each note

#for bird in bird_list:
#    if bird != 'w16w14': continue
#    print(bird)

#    df_bird = df[df['birdID'] == bird]

    # Get pre & post deafening
#    df_pre = df_bird[df_bird['taskName'] == 'Predeafening']

#    if df_pre.empty:
#        continue

    # Get pre-deafening baseline & last day
#    df_baseline = df_pre.groupby('note').mean().reset_index()


# SQL statement
query = f"""SELECT birdID, taskName, taskSessionPostDeafening, note, nbNoteUndir, entropyUndir, spectroTempEntropyUndir, entropyVarUndir  
FROM syllable_pcc 
WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit} 
AND taskName='Postdeafening'
"""
df_pcc = db.to_dataframe(query)

# Store values from the last day of neural recording in post-deafening in df_post
df_post = pd.DataFrame()

for bird in bird_list:

    df_bird = df_pcc[df_pcc['birdID'] == bird]
    df_temp = df_bird[df_bird['taskSessionPostDeafening'] == df_bird['taskSessionPostDeafening'].max()]
    total_nb_notes = df_temp.groupby(['birdID','note'])['nbNoteUndir'].sum().values  # Add a column with the number of each note
    df_temp = df_temp.groupby(['birdID','note']).mean().reset_index()  # get the mean feature value for the session
    df_temp['nbNoteUndir'] = total_nb_notes
    df_post = df_post.append(df_temp, ignore_index=True)

# Remove the notes if the number of notes is less than the criteria
df_post.drop(df_post[df_post['nbNoteUndir'] < nb_note_crit].index, inplace=True)
# Change column names to match pre-deafening
df_post.rename(columns={'entropyUndir':'entropy',
                        'spectroTempEntropyUndir': 'spectroTemporalEntropy',
                        'entropyVarUndir': 'entropyVar',
                        }, inplace=True)


# Plot results (df_baseline vs. df_post)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Loop through birds to plot all notes
dot_size = 60
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

for bird in bird_list:

    if bird == 'y44r34': continue
    print(bird)

    # bird = 'y44r34'
    df_bird_pre = df_baseline[df_baseline['birdID'] == bird]
    df_bird_post = df_post[df_post['birdID'] == bird]

    if df_bird_pre.empty or df_bird_post.empty:
        continue

    note_filter = df_bird_pre['note'].unique() == df_bird_post['note'].unique()
    common_notes = np.intersect1d(df_bird_post['note'].unique(), df_bird_pre['note'].unique())
    if len(df_bird_pre) > len(df_bird_post):
        df_bird_pre = df_bird_pre[note_filter]
    else:
        df_bird_post = df_bird_post[note_filter]

    if df_bird_pre.empty or df_bird_post.empty:
        continue

    sns.scatterplot(np.zeros(len(df_bird_pre)), df_bird_pre['entropy'],
                         s=dot_size, color='k', ax=ax)

    sns.scatterplot(np.ones(len(df_bird_post)) * df_bird_post['taskSessionPostDeafening'].unique(), df_bird_post['entropy'],
                         s=dot_size, color='k', ax=ax)
    #     ax.get_legend().remove()

    # # Connect two dots
    # for note in range(len(df_baseline)):
    #     plt.plot([0, df_last_day_mean['taskSessionPostDeafening'].unique()],
    #              [df_baseline['entropy'][note], df_last_day_mean['entropy'][note]],
    #              'k-', linewidth=1)

    # Connect two dots
    for note in df_bird_pre['note']:
        ax.plot([0, df_bird_post['taskSessionPostDeafening'].unique()],
                 [df_bird_pre[df_bird_pre['note']==note]['entropy'].values, df_bird_post[df_bird_post['note']==note]['entropy'].values],
                 'k-', linewidth=1)


ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
ax.set_ylim([0.5, 1])
ax.set_xlabel('Days after deafening')
plt.title('Pre-defeaning mean vs. Last day of neural recording after deafening', fontsize=8)
plt.show()




# Loop through birds to plot all notes
dot_size = 60
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

for bird in bird_list:

    if bird == 'y44r34': continue
    print(bird)

    # bird = 'y44r34'
    df_bird_pre = df_baseline[df_baseline['birdID'] == bird]
    df_bird_post = df_post[df_post['birdID'] == bird]

    if df_bird_pre.empty or df_bird_post.empty:
        continue

    note_filter = df_bird_pre['note'].unique() == df_bird_post['note'].unique()
    common_notes = np.intersect1d(df_bird_post['note'].unique(), df_bird_pre['note'].unique())
    if len(df_bird_pre) > len(df_bird_post):
        df_bird_pre = df_bird_pre[note_filter]
    else:
        df_bird_post = df_bird_post[note_filter]

    if df_bird_pre.empty or df_bird_post.empty:
        continue

    sns.scatterplot(np.zeros(len(df_bird_pre)), df_bird_pre['entropyVar'],
                         s=dot_size, color='k', ax=ax)

    sns.scatterplot(np.ones(len(df_bird_post)) * df_bird_post['taskSessionPostDeafening'].unique(), df_bird_post['entropyVar'],
                         s=dot_size, color='k', ax=ax)
    #     ax.get_legend().remove()

    # # Connect two dots
    # for note in range(len(df_baseline)):
    #     plt.plot([0, df_last_day_mean['taskSessionPostDeafening'].unique()],
    #              [df_baseline['entropy'][note], df_last_day_mean['entropy'][note]],
    #              'k-', linewidth=1)

    # Connect two dots
    for note in df_bird_pre['note']:
        ax.plot([0, df_bird_post['taskSessionPostDeafening'].unique()],
                 [df_bird_pre[df_bird_pre['note']==note]['entropyVar'].values,
                  df_bird_post[df_bird_post['note']==note]['entropyVar'].values],
                 'k-', linewidth=1)


ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
ax.set_ylim([0, 0.025])
ax.set_xlabel('Days after deafening')
plt.title('Pre-defeaning mean vs. Last day of neural recording after deafening', fontsize=8)
plt.show()