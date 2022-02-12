"""
Change the name of the .cbin & .rec files to fit the format (b14r74_190913_161407_Undir) used in the current analysis
"""


def main(data_path=None):
    import os
    from pathlib import Path
    from ..utils.functions import find_data_path

    if data_path is None:
        data_path = find_data_path()

    # Get .cbin, .rec files to process
    cbin_files = [str(file) for file in data_path.rglob('*.cbin')]
    rec_files = [str(file) for file in data_path.rglob('*.rec')]

    if not cbin_files:
        print('No .cbin files in the directory!')
    else:

        for cbin, rec in zip(cbin_files, rec_files):

            cbin_file = Path(cbin)  # e.g.,  'k27o36.3-03302012.188'
            rec_file = Path(rec)

            if not len(cbin_file.name.split('.')) == 2:  # not the file format sent from Mimi
                bird_id = cbin_file.stem.split('.')[0]
                date = cbin_file.stem.split('.')[1].split('-')[1]
                date = date[-2:] + date[:4]
                file_ind = cbin_file.stem.split('.')[-1]

                new_file_name = '_'.join([bird_id, date, file_ind])  # e.g., k27o36_120330_188
                new_cbin_file = new_file_name + Path(cbin).suffix  # add .cbin extension
                new_rec_file = new_file_name + Path(rec).suffix  # add .rec extension

                new_cbin_file = cbin_file.parent / new_cbin_file  # path
                new_rec_file = rec_file.parent / new_rec_file  # path

                if not new_cbin_file.exists():
                    print(cbin_file.name)
                    os.rename(cbin_file, new_cbin_file)
                    os.rename(rec_file, new_rec_file)

            else:
                print('Not the expected file format')
                break

        print('Done!')


if __name__ == '__main__':
    # Search for data dir manually
    main()
