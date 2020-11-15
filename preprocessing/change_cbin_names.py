"""
By Jaerong
Change the name of the .cbin files to fit the format (b14r74_190913_161407_Undir.rhd) used in the current analysis
"""

import os
from pathlib import Path
from utilities.functions import find_data_path


def main(data_path=None):

    if data_path is None:
        data_path = find_data_path()

    # Get .cbin files to process
    cbin_files = [str(rhd) for rhd in data_path.rglob('*.cbin')]

    if not cbin_files:
        print('No .cbin files in the directory!')
    else:

        for cbin in cbin_files:

            file = Path(cbin)  # e.g.,  'k27o36.3-03302012.188'

            if not len(file.name.split('.')) == 2:  # not the file format sent from Mimi
                bird_id = file.stem.split('.')[0]
                date = file.stem.split('.')[1].split('-')[1]
                date = date[-2:] + date[:4]
                file_ind = file.stem.split('.')[-1]

                new_file = '_'.join([bird_id, date, file_ind])  # e.g., k27o36_120330_188
                new_file += Path(cbin).suffix  # add .cbin extension
                new_file = file.parent / new_file  # path

                if not new_file.exists():
                    print(file.name)
                    os.rename(file, new_file)

                print('Done!')

            else:
                print('Not the expected file format')
                break


if __name__ == '__main__':

    # Specify data directory here

    data_path = Path(r'H:\Box\Data\Deafening Project\k27o36\Postdeafening\D28(20120330)\01\Songs')
    main(data_path)

    # Search for data dir manually
    # main()