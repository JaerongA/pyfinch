"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

import pathlib
from datetime import date
from database import load


def make_save_dir(dir_name, add_date=True):
    import os
    project_path = load.project()
    # print(project_path)

    save_path = project_path + '\\Analysis\\' + dir_name

    if add_date:
        today = date.today()
        save_path = save_path + '\\' + today.strftime("%Y-%m-%d")

    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)


def figure(path):
    pass


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)

