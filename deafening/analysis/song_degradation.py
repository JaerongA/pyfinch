"""
Compare song features before and after deafening
Post-deafening songs were obtained from the last session where neural data are present
"""

from analysis.parameters import fr_crit, nb_note_crit
from database.load import ProjectLoader
from util import save
import pandas as pd

features = ['entropy', 'spectroTemporalEntropy', 'entropyVar']

# Load database
db = ProjectLoader().load_db()
# SQL statement
# Select Undir & Predeafening
query = f"""SELECT syl.*, song.taskSessionPostDeafening, song.dph, song.block10days
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
query = f"""SELECT birdID, taskName, taskSessionDeafening, note, nbNoteUndir, frUndir,
pccUndir, entropyUndir, spectroTempEntropyUndir, entropyVarUndir  
FROM syllable_pcc 
WHERE frUndir >= {fr_crit} AND nbNoteUndir >={nb_note_crit} 
AND taskName='Postdeafening'
"""
df_pcc = db.to_dataframe(query)

for bird in bird_list:
    df_pcc_bird = df_pcc[df_pcc['birdID'] == bird_id]
    df_pcc_bird[df_pcc_bird['taskSessionPostDeafening'] == df_pcc_bird['taskSessionPostDeafening'].max()]


