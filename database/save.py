"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""


import pathlib
from datetime import date
from database import load

# print(project_path)
today = date.today()


def make_save_dir(path_name, add_date=True):
    project_path = load.project()
    # print(project_path)

    save_path = pathlib.Path(project_path + '\\Analysis\\' + path_name)

    if add_date:
        save_path = save_path + '\\' + today.strftime("%Y-%m-%d")

    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)


def file(path):
    pass


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    make_save_dir(path_name, date)
