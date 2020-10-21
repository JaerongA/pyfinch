"""
By Jaerong
In some sessions, the raw data were mistakenly loaded as ADBit values in Offline Sorter.
It significantly decreased the amplitude of the waveform of those clusters isolated under that setting.
This program converts the amplitude of those clusters by using ADbit value
"""


def convert_adbit2volts(spk_waveform):
    """Input the waveform matrix extracted from the cluster .txt output"""

    # Parameters on the Offline Sorter
    volt_range = 10  # in milivolts +- 5mV
    sampling_bits = 16
    volt_resolution = 0.000153  # volt_range/(2**sampling_bits)
    spk_waveform_new = spk_waveform / (0.000153 * 1E3)  # convert to microvolts
    return spk_waveform_new


def converter(query):
    """
    Input: SQL query
    Batch process from the database
    """
    from database import load
    import numpy as np
    from pathlib import Path

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


if __name__ == '__main__':

    query = "SELECT * FROM cluster WHERE adbit_cluster IS TRUE"
    converter(query)
