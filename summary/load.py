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

    # os.chdir(projectROOT)
    # print(os.listdir(summaryROOT))
    return projectROOT


def summary(parser):
    import os
    import pandas as pd

    summaryROOT = projectROOT + parser.get('folders', 'summaryROOT')
    summary_file = parser.get('files', 'summary')
    os.chdir(summaryROOT)
    # print(os.listdir(summaryROOT))
    summary_cluster = pd.read_excel(summary_file).applymap(str)  # read all cluster information as a string
    print('Loading the summary file')
    # print(summary_cluster.columns)  #  print out all features of the summary_cluster
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

    if len(cluster.TaskSession) == 1:
        cluster.TaskSession = 'D0' + cluster.TaskSession
    elif len(cluster.TaskSession) == 2:
        cluster.TaskSession = 'D' + cluster.TaskSession
    cluster.Site = cluster.Site[-2:]
    return cluster


def cluster_info(cluster):
    import os
    sessionID = cluster.Key + '-' + cluster.BirdID + '-' + cluster.TaskName + '-' + cluster.TaskSession + '-' + cluster.SessionDate + '-Site' + cluster.Site
    cellID = sessionID + '-' + cluster.Channel + '-' + cluster.Cluster
    cellROOT = projectROOT + '\\' + cluster.BirdID + '\\' + cluster.TaskName + '\\' + cluster.TaskSession + '(' + cluster.SessionDate + ')\\' + cluster.Site + '\\Songs'
    # print('Accessing... ' + cellROOT)
    # os.chdir(cellROOT)
    return sessionID, cellID, cellROOT


if __name__ == '__main__':
    config_file = '../project.ini'
    parser = config(config_file)
    projectROOT = project(parser)
    summary_cluster, nb_cluster = summary(parser)
