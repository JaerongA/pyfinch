from database import load
from analysis.spike import *
from analysis.parameters import *
from pathlib import Path
from analysis.load import read_rhd
import matplotlib.pyplot as plt
from util import save

query = "SELECT * FROM cluster WHERE id == 50 "
# query = "SELECT * FROM cluster WHERE ephysOK"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():
    ci = ClusterInfo(row)
    # # ci.name
    # # ci._load_spk()
    # # ci.waveform_analysis('rhd')
    # # ci._load_events()
    # # vars(ci)
    #
    # # dic = {}
    # # for col in row.keys():
    # #     dic[col]= row[col]
    #
    # # raw_data = RawData(row)
    # ci._load_events()
    # ci._load_spk()
    #
    #
    #
    #
    #
    # file_list = []
    # spk_list = []
    # onset_list = []
    # offset_list = []
    # syllable_list = []
    # duration_list = []
    # context_list = []
    #
    #
    # list_zip = zip(ci.files, ci.spk_ts, ci.onsets, ci.offsets, ci.syllables, ci.contexts)
    #
    # for file, spks, onsets, offsets, syllables, context in list_zip:
    #
    #     onsets = onsets.tolist()
    #     offsets = offsets.tolist()
    #
    #     # Find motifs
    #     motif_ind = find_str(ci.motif, syllables)
    #
    #     # Get syllable, analysis time stamps
    #     for ind in motif_ind:
    #         start_ind = ind
    #         stop_ind = ind + len(ci.motif) - 1
    #
    #         motif_onset = float(onsets[start_ind])
    #         motif_offset = float(offsets[stop_ind])
    #
    #         motif_spk = spks[np.where((spks >= motif_onset) & (spks <= motif_offset))]
    #         onsets_in_motif = onsets[start_ind:stop_ind+1]
    #         offsets_in_motif = offsets[start_ind:stop_ind+1]
    #
    #         # onsets_in_motif = [onset for onset in onsets if onset != '*' and motif_onset <= float(onset)  <= motif_offset]
    #         # offsets_in_motif = [offset for offset in offsets if offset != '*' and motif_onset <= float(offset)  <= motif_offset]
    #
    #         file_list.append(file)
    #         spk_list.append(motif_spk)
    #         duration_list.append(motif_offset-motif_onset)
    #         onset_list.append(onsets_in_motif)
    #         offset_list.append(offsets_in_motif)
    #         syllable_list.append(syllables[start_ind:stop_ind+1])
    #         context_list.append(context)

    # mi = MotifInfo(row)
    # isi = mi.isi