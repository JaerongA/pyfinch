"""
By Jaerong (2020/07/08)
A function that differentially labels undir and dir files by tagging the string at the file name (e.g., _Dir, _Undir)
"""

import os
import shutil
from tkinter import *
from tkinter import messagebox as mb

data_dir = r'H:\Box\Data\Deafening Project\y44r34\Predeafening\20200707\02'
os.chdir(data_dir)

list_dir = [dir for dir in os.listdir(data_dir)]

for tag in list_dir:

    new_path = os.path.join(data_dir, tag)
    os.chdir(new_path)
    list_file = [file for file in os.listdir(new_path)]

    for file in list_file:

        new_file = ''
        print('Processing... {}'.format(file))
        ext = os.path.splitext(file)[1]  # file extension
        if ext == '.txt':
            shutil.copyfile(file, os.path.join(data_dir, file))
        elif 'merged' in file:
            shutil.copyfile(file, os.path.join(data_dir, file))
        elif 'not' in file:  # .not.mat files
            new_file = '_'.join([file.split('.')[0], tag.title()]) + '.' + '.'.join(file.split('.')[-3:])
            shutil.copyfile(file, os.path.join(data_dir, new_file))
        elif 'labeling' in file:
            new_file = '_'.join([file[:str.find(file, '(labeling)')], tag.title()]) + file[
                                                                                      str.find(file, '(labeling)'):]
            shutil.copyfile(file, os.path.join(data_dir, new_file))
        else:
            new_file = '_'.join([file.split('.')[0], tag.title()]) + ext  # e.g., b70r38_190321_122233_Undir.rhd
            shutil.copyfile(file, os.path.join(data_dir, new_file))

os.chdir(data_dir)
root = Tk()
root.withdraw()


def ask():
    """ Ask if you wish to keep/delete the root folder"""
    resp = mb.askquestion('Question', 'Do you want to delete the root folder?')
    if resp == 'yes':
        shutil.rmtree(new_path)  # works even if the folder is not empty
        mb.showinfo('', 'Check the files!')
        root.destroy()
    else:
        mb.showinfo('', 'Done!')
        root.destroy()


ask()
