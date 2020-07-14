import os
from datetime import date
# from summary import load
# from load import config_file
# from summary.read_config import parser
from summary import load

project_path = load.project(parser)  # find the project folder
# print(project_path)
today = date.today()


def make_save_dir(path_name, *date):

    # print(path_name)

    save_root = project_path + '\\Analysis\\' + path_name
    if date:
        save_root = save_root + '\\' + today.strftime("%Y-%m-%d")
    if not os.path.exists(save_root):
        os.mkdir(save_root)


if __name__ == '__main__':
    make_save_dir(path_name, date)
