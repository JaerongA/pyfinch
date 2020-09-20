"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load
from load_intan_rhd_format.load_intan_rhd_format import read_data



query = "SELECT * FROM cluster WHERE id = '22'"
cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():

    cell_name, cell_path = load.cell_info(cell_info)
    # print(cell_name)
    print('Loading... ' + cell_name)


    # for file in [x for x in cell_path.iterdir() if x.is_dir()]:  # loop through the sub-dir
