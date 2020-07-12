# from tkinter import *
# from tkinter import messagebox as mb
#
# root = Tk()
# root.withdraw()
#
# def ask():
#     """ Ask if you wish to keep/delete the root folder"""
#     resp = mb.askquestion('Question', 'Do you want to delete the root folder?')
#     if resp == 'yes':
#         # os.rmdir(new_path)
#         mb.showinfo('', 'Check the files!')
#         root.destroy()
#         pass
#     else:
#         root.destroy()
#
#
# ask()


from summary import load

parser = load.config()
project_path = load.project(parser)
summary_cluster, nb_cluster = load.summary(parser)
print(summary_cluster, nb_cluster)