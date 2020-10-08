"""
By Jaerong
In some sessions, the raw data were mistakenly loaded as ADBit values in Offline sorter.
It significantly decreased the amplitude of the waveform of those clusters isolated under that setting.
This program converts the amplitude of those clusters by using ADbit value
"""



for row in cur.fetchall():
    cell_name, cell_path = load.cell_info(row)
    print('Loading... ' + cell_name)
    mat_file = list(cell_path.glob('*' + row['channel'] + '(merged).mat'))[0]
    channel_info = scipy.io.loadmat(mat_file)
    spk_file = list(cell_path.glob('*' + row['channel'] + '(merged).txt'))[0]
    unit_nb = int(row['unit'][-2:])

    # Extract the raw neural trace (from the .mat file)
    raw_trace = channel_info['amplifier_data'][0]

    # Read from the cluster .txt file
    spk_ts, spk_waveform, nb_spk = read_spk_txt(spk_file, unit_nb)
    if not row['adbit']:
        print('a')
    spk_waveform = spk_waveform / (10/65536 * 1E3)