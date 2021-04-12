"""
By Jaerong
Plot spike correlograms
"""

from contextlib import suppress

import matplotlib.pyplot as plt

from analysis.parameters import *
from analysis.spike import *
from database.load import ProjectLoader

# Parameter
normalize = False
update = False
nb_row = 7
nb_col = 3


def get_jittered_corr(ClassObject):
    correlogram_jitter = []

    for iter in range(shuffling_iter):
        ClassObject.jitter_spk_ts()
        corr_temp = ClassObject.get_correlogram(ClassObject.spk_ts_jittered, ClassObject.spk_ts_jittered)
        correlogram_jitter.append(corr_temp['U'])
    correlogram_jitter = np.array(correlogram_jitter)
    return correlogram_jitter


def get_corr_category(Correlogram, correlogram_jitter):
    corr_mean = correlogram_jitter.mean(axis=0)
    corr_std = correlogram_jitter.std(axis=0)
    upper_lim = corr_mean + (corr_std * 2)
    lower_lim = corr_mean - (corr_std * 2)
    # Check peak significance
    if Correlogram.peak_value > upper_lim[Correlogram.peak_ind]:
        category = 'Bursting'
    else:
        category = 'NonBursting'
    return category


# Load database
db = ProjectLoader().load_db()
# SQL statement
query = "SELECT * FROM cluster WHERE id = 96"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    correlogram = {}

    mi = MotifInfo(row, update=update)  # motif object
    bi = BaselineInfo(row, update=update)  # baseline object

    # Get correlogram
    correlogram = mi.get_correlogram(mi.spk_ts, mi.spk_ts)
    correlogram['B'] = bi.get_correlogram(bi.spk_ts, bi.spk_ts)

    # Get correlogram per condition
    with suppress(KeyError):

        corr_b = Correlogram(correlogram['B'])  # Load correlogram object
        corr_u = Correlogram(correlogram['U'])
        corr_d = Correlogram(correlogram['D'])

    correlogram_jitter = mi.get_jittered_corr()
    a = bi.get_jittered_corr()
    correlogram_jitter['B'] = bi.get_jittered_corr()
    # Combine correlogram from two contexts

    for key, value in correlogram_jitter.items():
        if key == 'U':
            corr_u.category(value)
        elif key == 'D':
            corr_d.category(value)
        elif key == 'B':
            corr_b.category(value)

    # Plot the results
    fig = plt.figure(figsize=(11, 6))
    fig.set_dpi(600)
    plt.suptitle(mi.name, y=.95)
    ax1 = plt.subplot2grid((nb_row, nb_col), (1, 0), rowspan=3, colspan=1)

    # For text output
    ax_txt1 = plt.subplot2grid((nb_row, nb_col), (4, 0), rowspan=3, colspan=1)

    with suppress(KeyError):

        corr_u.plot_corr(ax1, spk_corr_parm['time_bin'], correlogram['U'], 'Undir', normalize=normalize)
        # ax1.plot(spk_corr_parm['time_bin'], upper_lim, 'g', lw=0.5)

    plt.show()
    print(corr_u.category)
    print(corr_d.category)
    print(corr_b.category)
