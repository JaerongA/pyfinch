"""
Shuffle spikes with different parameters and see how the proportion of syllables having significant PCC changes
"""


import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from analysis.parameters import *
from analysis.spike import *
from database.load import DBInfo, ProjectLoader
from util import save
from util.draw import *

def create_db():
    from database.load import ProjectLoader

    db = ProjectLoader().load_db()
    with open('../../database/create_syllable_pcc_shuffle.sql', 'r') as sql_file:
        db.conn.executescript(sql_file.read())


# parameters
rec_yloc = 0.05
rec_height = 1  # syllable duration rect
text_yloc = 0.5  # text height
font_size = 12
marker_size = 0.4  # for spike count
nb_note_crit = 10  # minimum number of notes for analysis

norm_method = None
fig_ext = '.png'  # .png or .pdf
update = False  # Set True for recreating a cache file
save_fig = False
update_db = True  # save results to DB
time_warp = True  # spike time warping
shuffled_baseline = False
plot_hist = False

# Spike shuffling parameter for peth for getting baseline PCC
peth_shuffle = {'shuffle_limit': [1, 5, 10, 15, 20],  # in ms
                'shuffle_iter': 100}  # bootstrap iterations

# Create & Load database
if update_db:
    db = create_db()

# Load database
# SQL statement
# Create a new database (syllable)
db = ProjectLoader().load_db()
query = "SELECT * FROM cluster WHERE analysisOK=1 AND id>=87"
# query = "SELECT * FROM cluster WHERE analysisOK=1 AND id>=115"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    motif = cluster_db.motif

    # Load class object
    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object

    # Loop through note
    for note in cluster_db.songNote:

        ni = ci.get_note_info(note)
        if not ni:  # the target note does not exist
            continue

        # Skip if there are not enough motifs per condition
        if np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()]):
            print("Not enough notes")
            continue

        # GET PETH and firing rates
        pi = ni.get_note_peth()
        pi.get_fr()  # get firing rates

        # Calculate pairwise cross-correlation
        pi.get_pcc()

        # Save results to database
        if update_db:   # only use values from time-warped data
            query = "INSERT OR IGNORE INTO " \
                    "syllable_pcc_shuffle(clusterID, birdID, taskName, taskSession, taskSessionDeafening, taskSessionPostDeafening, dph, block10days, note)" \
                    "VALUES({}, '{}', '{}', {}, {}, {}, {}, {}, '{}')".format(cluster_db.id, cluster_db.birdID, cluster_db.taskName, cluster_db.taskSession,
                                                                              cluster_db.taskSessionDeafening, cluster_db.taskSessionPostDeafening,
                                                                              cluster_db.dph, cluster_db.block10days, note)
            db.cur.execute(query)

            if 'U' in ni.nb_note:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET nbNoteUndir = ({ni.nb_note['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'D' in ni.nb_note:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET nbNoteDir = ({ni.nb_note['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'U' in ni.mean_fr and ni.nb_note['U'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET frUndir = ({ni.mean_fr['U']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
            if 'D' in ni.mean_fr and ni.nb_note['D'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET frDir = ({ni.mean_fr['D']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'U' in pi.pcc and ni.nb_note['U'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET pccUndir = ({pi.pcc['U']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            if 'D' in pi.pcc and ni.nb_note['D'] >= nb_note_crit:
                db.cur.execute(f"UPDATE syllable_pcc_shuffle SET pccDir = ({pi.pcc['D']['mean']}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")

            db.conn.commit()


        from collections import defaultdict
        from functools import partial
        import scipy.stats as stats

        # One-sample t-test (one-sided)
        alpha = 0.05
        p_val_dict = defaultdict(list)
        p_sig_dict = defaultdict(list)

        pcc_shuffle = defaultdict(partial(np.ndarray, 0))

        for ind, shuffle_limit in enumerate(peth_shuffle['shuffle_limit']):
            for iter in range(peth_shuffle['shuffle_iter']):
                ni.jitter_spk_ts(shuffle_limit)
                pi_shuffle = ni.get_note_peth(shuffle=True)  # peth object
                pi_shuffle.get_fr()  # get firing rates
                pi_shuffle.get_pcc()  # get pcc
                for context, pcc in pi_shuffle.pcc.items():
                    pcc_shuffle[context] = np.append(pcc_shuffle[context], pcc['mean'])

            for context in pcc_shuffle.keys():
                p_val = stats.ttest_1samp(a=pcc_shuffle[context], popmean=pi.pcc[context]['mean'],
                                          nan_policy='omit', alternative='less')[1]
                p_val_dict[context].append(p_val)
                p_sig_dict[context].append(p_val < alpha)


        if update_db:   # only use values from time-warped data

            for ind, shuffle_limit in enumerate(peth_shuffle['shuffle_limit']):
                if f'pccUndirSig_{shuffle_limit}' not in db.col_names('syllable_pcc_shuffle'):
                    db.cur.execute(f"ALTER TABLE syllable_pcc_shuffle ADD COLUMN pccUndirSig_{shuffle_limit} BOOLEAN")
                if f'pccDirSig_{shuffle_limit}' not in db.col_names('syllable_pcc_shuffle'):
                    db.cur.execute(f"ALTER TABLE syllable_pcc_shuffle ADD COLUMN pccDirSig_{shuffle_limit} BOOLEAN")

                if 'U' in f'pccUndirSig_{shuffle_limit}' and ni.nb_note['U'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_shuffle SET pccUndirSig_{shuffle_limit} = ({p_sig_dict['U'][ind]}) WHERE clusterID = {cluster_db.id} AND note = '{note}'")
                if 'D' in f'pccDirSig_{shuffle_limit}' and ni.nb_note['D'] >= nb_note_crit:
                    db.cur.execute(f"UPDATE syllable_pcc_shuffle SET pccDirSig_{shuffle_limit} = ({p_sig_dict['D'][ind]})  WHERE clusterID = {cluster_db.id} AND note = '{note}'")
            db.conn.commit()

        # Plot pcc shuffle histogram for verification
        # plt.hist(pi.pcc['D']['array'], density=True)
        # plt.hist(pcc_shuffle['D'], density=True)
        # plt.axvline(pi.pcc['D']['mean'], color='k')
        # # plt.legend('original', 'shuffled')
        # plt.show()


# Convert db to csv
if update_db:
    db.to_csv('syllable_pcc_shuffle')
print('Done!')
