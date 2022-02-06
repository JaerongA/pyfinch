"""
Get correlation between entropy and number of spikes per syllable
v2 computes values from 50 ms prior to onset to 50 ms prior to offset
"""

from analysis.parameters import note_buffer, freq_range, nb_note_crit, pre_motor_win_size, alpha
from analysis.spike import ClusterInfo, AudioData
from database.load import ProjectLoader, DBInfo
from util import save
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from util.draw import remove_right_top
import pandas as pd

# Parameter
update = False  # update or make a new cache file
save_spectrogram = False
save_result_fig = True  # correlation figure
view_folder = False  # view the folder where figures are stored
update_db = True  # save results to DB
save_csv = True
fig_ext = '.png'  # .png or .pdf
txt_offset = 0.2
correlation_parm = 'ev'  # {'spectral_entropy' : 'spectro_temporal_entropy' : 'ev'}

# Load database
db = ProjectLoader().load_db()

# Make database
with open('../database/create_entropy_spk_corr.sql', 'r') as sql_file:
    db.conn.executescript(sql_file.read())

# Make save path
save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'EntropySpkCorrNew', add_date=False)

# SQL statement
# query = "SELECT * FROM cluster WHERE id=96"
# query = "SELECT * FROM cluster WHERE analysisOK AND id=96"
query = "SELECT * FROM cluster WHERE analysisOK"
db.execute(query)

# Loop through db
for row in db.cur.fetchall():

    # Load cluster info from db
    cluster_db = DBInfo(row)
    name, path = cluster_db.load_cluster_db()
    unit_nb = int(cluster_db.unit[-2:])
    channel_nb = int(cluster_db.channel[-2:])
    format = cluster_db.format
    song_note = cluster_db.songNote

    # Load class object
    ci = ClusterInfo(path, channel_nb, unit_nb, format, name, update=update)  # cluster object
    audio = AudioData(path, update=update)  # audio object

    for note in cluster_db.songNote:
        if update_db:
            query = f"INSERT OR IGNORE INTO entropy_spk_corr (clusterID, birdID, taskName, taskSession, note) " \
                    f"VALUES({cluster_db.id}, '{cluster_db.birdID}', '{cluster_db.taskName}', {cluster_db.taskSession}, '{note}')"
            db.cur.execute(query)
            db.conn.commit()

    # Load if cvs that has calculated FF and spike count already exists
    csv_name = ci.name + '.csv'
    csv_path = save_path / csv_name
    if csv_path.exists() and not save_spectrogram:
        df = pd.read_csv(csv_path)
    else:
        # Calculate entropy
        df = pd.DataFrame()  # Store results here
        for note in cluster_db.songNote:
            # Load note object
            ni = ci.get_note_info(note)

            if not ni or (np.prod([nb[1] < nb_note_crit for nb in ni.nb_note.items()])):  # if the note does not exist
                db.cur.execute(f"DELETE FROM entropy_spk_corr WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                db.conn.commit()
                continue

            zipped_lists = zip(ni.onsets, ni.offsets, ni.contexts, ni.spk_ts)
            for note_ind, (onset, offset, context, spk_ts) in enumerate(zipped_lists):

                # Loop through the notes
                # if note_ind != 1:
                #     continue
                duration = offset - onset

                # Get spectrogram
                # Load audio object with info from .not.mat files
                # timestamp, data = audio.extract([onset, offset])
                timestamp, data = audio.extract([onset, offset])
                spect_time, spect, spect_freq = audio.spectrogram(timestamp, data)
                spectral_entropy = audio.get_spectral_entropy(spect, mode='spectral')
                se_dict = audio.get_spectral_entropy(spect, mode='spectro_temporal')

                if save_spectrogram:
                    # Plot figure
                    font_size = 6
                    fig = plt.figure(figsize=(4, 3), dpi=250)
                    fig_name = f"#{note_ind :03} - {note} - {context}"

                    plt.suptitle(fig_name, y=.90, fontsize=font_size)
                    gs = gridspec.GridSpec(4, 6)

                    # Plot spectrogram
                    ax_spect = plt.subplot(gs[1:3, 0:3])
                    spect_time = spect_time - spect_time[0]  # starts from zero
                    ax_spect.pcolormesh(spect_time, spect_freq, spect,  # data
                                        cmap='hot_r',
                                        norm=colors.SymLogNorm(linthresh=0.05,
                                                               linscale=0.03,
                                                               vmin=0.5,
                                                               vmax=100
                                                               ))

                    remove_right_top(ax_spect)
                    ax_spect.set_xlim(-note_buffer, duration + note_buffer)
                    ax_spect.set_ylim(freq_range[0], freq_range[1])
                    ax_spect.set_xlabel('Time (ms)', fontsize=font_size)
                    ax_spect.set_ylabel('Frequency (Hz)', fontsize=font_size)
                    plt.yticks(freq_range, list(map(str, freq_range)), fontsize=5)
                    plt.xticks(fontsize=5), plt.yticks(fontsize=5)

                    # Calculate spectral entropy per time bin
                    # Plot syllable entropy
                    ax_se = ax_spect.twinx()
                    ax_se.plot(spect_time, se_dict['array'], 'k')
                    ax_se.set_ylim(0, 1)
                    ax_se.spines['top'].set_visible(False)
                    ax_se.set_ylabel('Entropy', fontsize=font_size)
                    plt.xticks(fontsize=5), plt.yticks(fontsize=5)

                    # Print out text results
                    txt_xloc = -1.5
                    txt_yloc = 0.8
                    ax_txt = plt.subplot(gs[1:, -1])
                    ax_txt.set_axis_off()  # remove all axes
                    ax_txt.text(txt_xloc, txt_yloc, f"Spectral Entropy = {round(spectral_entropy, 3)}",
                                fontsize=font_size)
                    txt_yloc -= txt_offset
                    ax_txt.text(txt_xloc, txt_yloc, f"Spectrotemporal Entropy = {round(se_dict['mean'], 3)}",
                                fontsize=font_size)
                    txt_yloc -= txt_offset
                    ax_txt.text(txt_xloc, txt_yloc, f"Entropy Variance = {round(se_dict['var'], 4)}",
                                fontsize=font_size)

                    # Save figure
                    save_path2 = save.make_dir(save_path / ci.name, note, add_date=False)
                    save_path3 = save.make_dir(save_path, correlation_parm + 'new_window', add_date=False)  # save .csv and result figures
                    save.save_fig(fig, save_path2, fig_name, view_folder=view_folder, fig_ext=fig_ext)

                # Get number of spikes from pre-motor window
                # pre_motor_win = [onset - pre_motor_win_size, onset]
                pre_motor_win = [onset - pre_motor_win_size, offset - pre_motor_win_size]
                nb_spk = len(spk_ts[np.where((spk_ts >= pre_motor_win[0]) & (spk_ts <= pre_motor_win[-1]))])

                # Organize results per song session
                temp_df = pd.DataFrame({'note': [note],
                                        'context': [context],
                                        'spectral_entropy': [round(spectral_entropy, 3)],
                                        'spectro_temporal_entropy': [round(se_dict['mean'], 3)],
                                        'ev': [round(se_dict['var'], 4)],
                                        'nb_spk': [nb_spk]
                                        })
                df = df.append(temp_df, ignore_index=True)

        # Save ff and premotor spikes of each syllable rendition to csv
        if save_csv:
            df.index.name = 'Index'
            csv_name = ci.name + '.csv'
            save_path3 = save.make_dir(save_path, correlation_parm + ' new_window', add_date=False)  # save .csv and result figures
            df.to_csv(save_path3 / csv_name, index=True, header=True)  # save the dataframe to .cvs format

    # Draw correlation scatter between spk count and FF
    if not df.empty:
        font_size = 15
        for i, note in enumerate(sorted(df['note'].unique())):
            for context in sorted(df['context'].unique(), reverse=True):
                temp_df = df[(df['note'] == note) & (df['context'] == context)]

                # Go ahead with calculation only if it meets the number of notes criteria
                if len(temp_df) >= nb_note_crit:

                    if 'fig2' not in locals():
                        fig2 = plt.figure(figsize=(len(set(df['note'])) * 6, len(set(df['context'].unique())) * 6))
                        plt.suptitle(ci.name, y=.9, fontsize=font_size)

                    # Calculate values
                    pre_motor_fr = round(
                        temp_df['nb_spk'].sum() / (len(temp_df['nb_spk']) * (pre_motor_win_size / 1E3)),
                        3)  # firing rates during the pre-motor window
                    corr, corr_pval = pearsonr(temp_df['nb_spk'], temp_df[correlation_parm])
                    pval_sig = True if corr_pval < alpha else False
                    r_square = corr ** 2
                    polarity = 'positive' if corr > 0 else 'negative'

                    def get_shuffled_sig_prop():
                        shuffle_iter = 100
                        sig_array = np.array([], dtype=np.int)
                        for i in range(shuffle_iter):
                            shuffle_df = temp_df
                            corr, corr_pval = pearsonr(temp_df['nb_spk'], shuffle_df[correlation_parm].sample(frac=1))
                            pval_sig = True if corr_pval < alpha else False
                            sig_array = np.append(sig_array, pval_sig)
                        return sig_array

                    shuffled_sig_prop = get_shuffled_sig_prop()

                    if len(df['context'].unique()) == 2 and context == 'U':
                        ax = plt.subplot2grid((13, len(set(df['note'])) + 1), (1, i), rowspan=3, colspan=1)
                    elif len(df['context'].unique()) == 2 and context == 'D':
                        ax = plt.subplot2grid((13, len(set(df['note'])) + 1), (8, i), rowspan=3, colspan=1)
                    else:
                        ax = plt.subplot2grid((7, len(set(df['note'])) + 1), (1, i), rowspan=3, colspan=1)

                    ax.scatter(temp_df['nb_spk'], temp_df[correlation_parm], color='k', s=5)
                    # ax.scatter(temp_df['nb_spk'], temp_df['spectro_temporal_entropy'], color='k', s=5)
                    ax.set_title(f"Note ({note}) - {context}", size=font_size)
                    if i == 0:
                        if correlation_parm == 'spectro_temporal_entropy':
                            ax.set_ylabel('Spectrotemporal Entropy')
                        elif correlation_parm == 'spectral_entropy':
                            ax.set_ylabel('Spectral Entropy')
                        elif correlation_parm == 'ev':
                            ax.set_ylabel('Entropy Variance')

                    ax.set_xlabel('Spk Count')
                    ax.set_xlim([-0.5, temp_df['nb_spk'].max() + 1])
                    remove_right_top(ax)

                    # Print out results
                    if len(df['context'].unique()) == 2 and context == 'U':
                        ax_txt = plt.subplot2grid((13, len(set(df['note'])) + 1), (4, i), rowspan=1, colspan=1)
                    elif len(df['context'].unique()) == 2 and context == 'D':
                        ax_txt = plt.subplot2grid((13, len(set(df['note'])) + 1), (11, i), rowspan=1, colspan=1)
                    else:
                        ax_txt = plt.subplot2grid((7, len(set(df['note'])) + 1), (4, i), rowspan=1, colspan=1)

                    txt_xloc = 0
                    txt_yloc = 0
                    txt_inc = 0.6

                    ax_txt.text(txt_xloc, txt_yloc, f"nbNotes = {len(temp_df)}", fontsize=font_size)
                    txt_yloc -= txt_inc
                    ax_txt.text(txt_xloc, txt_yloc, f"PremotorFR = {round(pre_motor_fr, 3)} (Hz)", fontsize=font_size)
                    txt_yloc -= txt_inc
                    ax_txt.text(txt_xloc, txt_yloc, f"CorrR = {round(corr, 3)}", fontsize=font_size)
                    txt_yloc -= txt_inc
                    t = ax_txt.text(txt_xloc, txt_yloc, f"CorrR Pval = {round(corr_pval, 3)}", fontsize=font_size)
                    if corr_pval < alpha:
                        corr_sig = True
                        t.set_bbox(dict(facecolor='green', alpha=0.5))
                    else:
                        corr_sig = False
                        t.set_bbox(dict(facecolor='red', alpha=0.5))

                    txt_yloc -= txt_inc
                    ax_txt.text(txt_xloc, txt_yloc, f"R_square = {round(r_square, 3)}", fontsize=font_size)
                    ax_txt.axis('off')

                if update_db:
                    if context == 'U':
                        db.cur.execute(f"UPDATE entropy_spk_corr SET nbNoteUndir={len(temp_df)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                        if len(temp_df) >= nb_note_crit:
                            db.cur.execute(f"UPDATE entropy_spk_corr SET premotorFRUndir={pre_motor_fr} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET entropyUndir={temp_df['spectral_entropy'].mean() : .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spectroTemporalEntropyUndir={temp_df['spectro_temporal_entropy'].mean(): .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET entropyVarUndir={temp_df['ev'].mean(): .4f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrSigUndir={pval_sig} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrUndir={round(corr, 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrRsquareUndir={round(r_square, 5)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET shuffledSigPropUndir={round(shuffled_sig_prop.mean(), 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                    elif context == 'D':
                        db.cur.execute(f"UPDATE entropy_spk_corr SET nbNoteDir={len(temp_df)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                        if len(temp_df) >= nb_note_crit:
                            db.cur.execute(f"UPDATE entropy_spk_corr SET premotorFRDir={pre_motor_fr} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET entropyDir={temp_df['spectral_entropy'].mean() : .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spectroTemporalEntropyDir={temp_df['spectro_temporal_entropy'].mean(): .3f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET entropyVarDir={temp_df['ev'].mean(): .4f} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrSigDir={pval_sig} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrDir={round(corr, 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET spkCorrRsquareDir={round(r_square, 5)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                            db.cur.execute(f"UPDATE entropy_spk_corr SET shuffledSigPropDir={round(shuffled_sig_prop.mean(), 3)} WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                    db.conn.commit()

            if update_db:
                # If neither condition meets the number of notes criteria
                db.cur.execute(f"SELECT nbNoteUndir, nbNoteDir FROM entropy_spk_corr WHERE clusterID={cluster_db.id} AND note= '{note}'")
                nb_notes = [{'U': data[0], 'D': data[1]} for data in db.cur.fetchall()][0]
                if not (bool(nb_notes['U']) or bool(nb_notes['D'])):
                    db.cur.execute(f"DELETE FROM entropy_spk_corr WHERE clusterID= {cluster_db.id} AND note= '{note}'")
                db.conn.commit()

    if save_result_fig:
        save_path3 = save.make_dir(save_path, correlation_parm + ' new_window', add_date=False)  # save .csv and result figures
        save.save_fig(fig2, save_path3, ci.name, view_folder=False, fig_ext=fig_ext)
    del fig2

if update_db:
    db.to_csv(f'entropy_spk_corr')

print('Done!')

