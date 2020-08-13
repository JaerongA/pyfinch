import os
from datetime import date
from summary import load

# print(project_path)
today = date.today()


def make_save_dir(path_name, *date):

    # print(path_name)

    save_dir = project_path + '\\Analysis\\' + path_name
    if date:
        save_dir = save_dir + '\\' + today.strftime("%Y-%m-%d")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)



def save_bout(filename, data):
    # save the song bout & number of bouts in .json
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    make_save_dir(path_name, date)
