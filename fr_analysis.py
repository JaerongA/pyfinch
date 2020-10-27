"""
By Jaerong
Run firing rate analysis
Get mean firing rates per condition
"""


from database import load
import numpy as np
from pathlib import Path

# query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
query = "SELECT * FROM cluster WHERE ephysOK IS TRUE"
cur, conn, col_names = load.database(query)

for row in cur.fetchall():
    cell_name, cell_path = load.cell_info(row)
    print('Loading... ' + cell_name)
    unit_nb = int(row['unit'][-2:])

    # Read from the cluster .txt file
    spk_txt_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]

    # Get the header
    f = open(spk_txt_file, 'r')
    header = f.readline()[:-1]

    spk_info = np.loadtxt(spk_txt_file, delimiter='\t', skiprows=1)  # skip header
    spk_waveform = spk_info[:, 3:]  # spike waveform

    # Convert the value
    spk_waveform_new = convert_adbit2volts(spk_waveform)
    spk_txt_file_new = Path(spk_txt_file.parent, f"{spk_txt_file.stem}_new{spk_txt_file.suffix}")

    # Replace the waveform  with new values
    spk_info[:, 3:] = spk_waveform_new

    # Save to a new cluster .txt file
    np.savetxt(spk_txt_file_new, spk_info, delimiter='\t', header=header, comments='', fmt='%f')
