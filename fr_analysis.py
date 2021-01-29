"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""

from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
from util.functions import *

# Parameters
update = False

# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # ci = ClusterInfo(row, update=update)
    bi = BaselineInfo(row, update=update)
    mi = MotifInfo(row, update=update)
    print(bi.mean_fr)
    print(mi.mean_fr['D'])
    print(mi.mean_fr['U'])

    # nb_spk = mi.spk_ts[0].shape[0]
    # np.random.seed(1)  #  make random jitter reproducible
    # jitter = np.random.uniform(-jitter_limit, jitter_limit, nb_spk)
    # spk_jittered = mi.spk_ts[0] + jitter
    # print(spk_jittered)
    #
    # spk_ts_jittered_list = []
    #
    # for ind, spk_ts in enumerate(mi.spk_ts):
    #     np.random.seed(ind)  # make random jitter reproducible
    #     nb_spk = spk_ts.shape[0]
    #     jitter = np.random.uniform(-jitter_limit, jitter_limit, nb_spk)
    #     spk_ts_jittered_list.append(spk_ts + jitter)


    mi.jitter_spk_ts()



    break
