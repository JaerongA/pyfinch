""" a function that differentially labels undir and dir files by tagging the string at the file name (e.g., _Dir, _Undir) """
import os


data_path = r'C:\Users\jahn02\Box\Data\Deafening Project\y44r34\Predeafening\20200707\01'
os.chdir(data_path)


list_folder = [dir for dir in os.listdir(data_path)]

for dir in list_folder:

    new_path = os.path.join(data_path, dir)
    list_file = [file for file in os.listdir(new_path)]

    for file in list_file:
        print(file)


