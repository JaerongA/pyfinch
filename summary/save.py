import os
from datetime import date
# from summary import load
# from load import config_file
from summary.read_config import parser
from summary import load

projectROOT = load.project(parser)  # find the project folder


print(projectROOT)
path_name = 'Information'
today = date.today()
# d = today.strftime("%Y-%m-%d")


def make_save_dir(path_name, *date):

    # print(path_name)

    saveROOT = projectROOT + '\\Analysis\\' + path_name
    if date:
        saveROOT = saveROOT + '\\' + today.strftime("%Y-%m-%d")
    if not os.path.exists(saveROOT):
        os.mkdir(saveROOT)


if __name__ == '__main__':
    make_save_dir(path_name, date)
