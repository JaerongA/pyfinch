"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from database import load
from load_intan_rhd_format.load_intan_rhd_format import read_data


query = "SELECT * FROM cluster WHERE birdID = 'g35r38'"

cur, conn, col_names = load.database(query)

for cell_info in cur.fetchall():

    cell_name, cell_path = load.cell_info(cell_info)
    # print(cell_name)
    print(cell_path)


