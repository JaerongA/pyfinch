# demarcate bout
# extract audio, neural data (by using class)

from analysis.spike import *
from util import save
from util.draw import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

# Parameters
save_fig = True
update = False
dir_name = 'RasterBouts'
fig_ext = '.png'  # .png or .pdf

# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM cluster"
# query = "SELECT * FROM cluster WHERE ephysOK"
query = "SELECT * FROM cluster WHERE id = 6"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    name, path = load_cluster(row)
    unit_nb = int(row['unit'][-2:])
    channel_nb = int(row['channel'][-2:])
    format = row['format']

    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    nd = NeuralData(path, channel_nb, format, update=update)  # raw neural data
    audio = AudioData(path, update=update)


    # # Motif start and end
    # start = onset[0] - peth_parm['buffer']
    # end = offset[-1] + peth_parm['buffer']
    # duration = offset[-1] - onset[0]
    #
    #
    # # Get spectrogram
    # audio = AudioData(row).extract([start, end])
    # audio.spectrogram(freq_range=freq_range)
    #
    # # Plot figure
    # fig = plt.figure(figsize=(8, 9), dpi=800)
    # # Plot spectrogram
    # ax_spect = plt.subplot(gs[1:3, 0:4])
    # ax_spect.pcolormesh(audio.timebins * 1E3 - peth_parm['buffer'], audio.freqbins, audio.spect,  # data
    #                     cmap='hot_r',
    #                     norm=colors.SymLogNorm(linthresh=0.05,
    #                                            linscale=0.03,
    #                                            vmin=0.5,
    #                                            vmax=100
    #                                            ))
    #
    # remove_right_top(ax_spect)
    # ax_spect.set_xlim(-peth_parm['buffer'], duration + peth_parm['buffer'])
    # ax_spect.set_ylim(freq_range[0], freq_range[1])
    # ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
    # plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
    # plt.setp(ax_spect.get_xticklabels(), visible=False)