# Get raster & peth

from database.load import ProjectLoader
from analysis.spike import *
from analysis.parameters import *
from scipy.io import wavfile
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from util import save
from util.spect import *
from util.draw import *
import math
from scipy.ndimage import gaussian_filter1d


query = "SELECT * FROM cluster WHERE id == 96"
# query = "SELECT * FROM cluster WHERE ephysOK"

project = ProjectLoader()
cur, conn, col_names = project.load_db(query)




# def get_peth(evt_ts_list, spk_ts_list):
#     """Get peri-event histogram & firing rates"""
#
#     import math
#
#     peth = np.zeros((len(evt_ts_list), peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of trials x nb of time bins
#
#     for trial_ind, (evt_ts, spk_ts) in enumerate(zip(evt_ts_list, spk_ts_list)):
#
#         evt_ts = np.asarray(list(map(float, evt_ts)))
#
#         spk_ts -= evt_ts[0]
#
#         for spk in spk_ts:
#             ind = math.ceil(spk / peth_parm['bin_size'])
#             print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
#             peth[trial_ind, ind] += 1
#     return peth



for row in cur.fetchall():


    cluster = DBLoader(row)

    ci = ClusterInfo(row)
    ci.load_events()
    mi = MotifInfo(row, update=True)

    # peth = np.zeros((len(mi),peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of motifs x nb of time bins
    # peth_warp = np.zeros((len(mi),peth_parm['bin_size'] * peth_parm['nb_bins']))  # nb of motifs x nb of time bins
    #
    # # Plot spectrogram & peri-event histogram
    # list_zip = zip(mi.onsets, mi.offsets, mi.spk_ts, mi.spk_ts_warp)
    # for motif_ind, (onset, offset, spk_ts, spk_ts_warp) in enumerate(list_zip):
    #
    #     # Convert from string to array of floats
    #     onset = np.asarray(list(map(float, onset)))
    #     offset = np.asarray(list(map(float, offset)))
    #
    #     spk_train = spk_ts - onset[0]
    #     spk_train_warp = spk_ts_warp - onset[0]
    #
    #     for spk in spk_train:
    #         ind = math.ceil(spk / peth_parm['bin_size'])
    #         print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
    #         peth[motif_ind, ind] += 1
    #
    #     for spk in spk_train_warp:
    #         ind = math.ceil(spk / peth_parm['bin_size'])
    #         print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
    #         peth_warp[motif_ind, ind] += 1

    # peth = get_peth(mi.onsets, mi.spk_ts)

    # pi = mi.get_peth()
    pi = mi.get_peth()

    pi.fr_smoothed = gaussian_filter1d(pi.fr, 8)

    # len(pi)

# a = np.asarray(mi.contexts)
# ind = np.where(a=='U')
# peth[ind]
plt.plot(pi.time_bin, pi.fr,'k')
plt.plot(pi.time_bin, pi.fr_smoothed,'r')
# plt.plot()
plt.show()