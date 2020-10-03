"""
By Jaerong
Calculates a spike signal-to-noise ratio (SNR) relative to the background
"""

from pathlib import Path
from datetime import date
from database import load


def make_save_dir(dir_name, add_date=True):
    import os
    project_path = load.project()

    # save_path = project_path + '/Analysis/' + dir_name
    save_path = Path(project_path) / 'Analysis' / dir_name

    if add_date:
        today = date.today()
        save_path = save_path / today.strftime("%Y-%m-%d")
        # save_path = Path(save_path '/' today.strftime("%Y-%m-%d"))

    print(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    return save_path


def figure(fig, save_path, title, ext='.png'):
    import matplotlib.pyplot as plt
    fig_name = Path(save_path + '/' + title + ext)
    plt.savefig(fig_name, transparent=True)

    pass


def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)
