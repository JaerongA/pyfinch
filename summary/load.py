## Load project & summary folder summary file
config_file = '../project.ini'


def config(config_file):
    from configparser import ConfigParser
    parser = ConfigParser()
    parser.read(config_file)
    # print(parser.sections())
    return parser


def project(parser):
    global projectROOT, summaryROOT

    projectROOT = parser.get('folders', 'projectROOT')
    summaryROOT = projectROOT + parser.get('folders', 'summaryROOT')

    # os.chdir(projectROOT)
    # print(os.listdir(summaryROOT))
    return projectROOT


def summary(parser):
    import os
    import pandas as pd

    summary_file = parser.get('files', 'summary')
    os.chdir(summaryROOT)
    # print(os.listdir(summaryROOT))
    summary_cluster = pd.read_excel(summary_file).applymap(str)  # read all cluster information as a string
    print('Loading the summary file')
    nb_cluster = summary_cluster.shape[0]  # nb of clusters
    return summary_cluster, nb_cluster


def cluster(summary_cluster, cluster_run):
    from types import SimpleNamespace
    this_dic = summary_cluster.iloc[cluster_run].to_dict()
    cluster = SimpleNamespace(**this_dic)

    if len(cluster.Key) == 1:
        cluster.Key = '00' + cluster.Key
    elif len(cluster.Key) == 2:
        cluster.Key = '0' + cluster.Key

    # if len(TaskSession) == 1: TaskSession = 'D0' + TaskSession
    # elif len(TaskSession) == 2: TaskSession = 'D' + TaskSession


    return cluster


def cluster_info(cluster):
    pass


if __name__ == '__main__':
    parser = load_ini(config_file)
    projectROOT, summaryROOT = load_project(parser)
    summary_cluster = load_cluster(parser)
