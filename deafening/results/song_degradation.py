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

# Song features to compare
features = ['entropy', 'spectroTemporalEntropy', 'entropyVar']

# Load database
db = ProjectLoader().load_db()

# Get pre-deafening baseline
query = f"""SELECT syl.*, song.taskSessionPostDeafening
        FROM individual_syllable syl INNER JOIN song 
        ON song.id = syl.songID WHERE syl.context='U' AND syl.taskName='Predeafening'"""
df = db.to_dataframe(query)

bird_list = df['birdID'].unique()

df_baseline = df.groupby(['birdID','note']).mean().reset_index()  # Get average feature values per bird per note
df_baseline['nb_notes'] = df.groupby(['birdID','note'])['note'].count().values  # Add a column with the number of each note
df_baseline.rename(columns={'nb_notes' : 'nbNotePre',
                        'entropy':'entropyPre',
                        'spectroTemporalEntropy': 'spectroTemporalEntropyPre',
                        'entropyVar': 'entropyVarPre',
                        }, inplace=True)
df_baseline.drop('taskSessionPostDeafening', axis=1, inplace=True)


# Get post-deafening where neural data are present
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
df_post.rename(columns={'nbNoteUndir' : 'nbNotePost',
                        'entropyUndir':'entropyPost',
                        'spectroTempEntropyUndir': 'spectroTemporalEntropyPost',
                        'entropyVarUndir': 'entropyVarPost',
                        }, inplace=True)

# Plot results
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Loop through birds to plot all notes
dot_size = 60
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
nb_notes = 0 # total number of notes plotted (that has both pre and post data)

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
    sns.scatterplot(np.zeros(len(df_merge)), df_merge['entropyPre'],
                         s=dot_size, color='k', ax=ax)

    sns.scatterplot(np.ones(len(df_merge)) * df_merge['taskSessionPostDeafening'].unique(), df_merge['entropyPost'],
                         s=dot_size, color='k', ax=ax)

    # Connect two dots
    for note in df_merge['note']:
        ax.plot([0, df_merge['taskSessionPostDeafening'].unique()],
                 [df_merge[df_merge['note']==note]['entropyPre'].values,
                  df_merge[df_merge['note']==note]['entropyPost'].values],
                 'k-', linewidth=1)

ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
ax.set_ylim([0.3, 1])
ax.set_xlabel('Days after deafening')
ax.set_ylabel('Entropy')
plt.suptitle(f"""Pre-defeaning mean vs. Last day of neural recording after deafening
                \n # of notes = {nb_notes}""", y=0.9, fontsize=8)
plt.show()