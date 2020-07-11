from tkinter import *
from tkinter import messagebox as mb

root = Tk()
root.withdraw()

def ask():
    """ Ask if you wish to keep/delete the root folder"""
    resp = mb.askquestion('Question', 'Do you want to delete the root folder?')
    if resp == 'yes':
        # os.rmdir(new_path)
        mb.showinfo('', 'Check the files!')
        root.destroy()
        pass
    else:
        root.destroy()


ask()
