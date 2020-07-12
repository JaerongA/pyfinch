"""By Jaerong
Load all info relating to the project """

# Load project & summary folder summary file


def config():
    from configparser import ConfigParser
    config_file = 'summary/project.ini'
    parser = ConfigParser()
    parser.read(config_file)
    # print(parser.sections())
    return parser


def project(parser):
    global project_path, summary_path

    project_path = parser.get('folders', 'project_path')

    # os.chdir(project_path)
    # print(os.listdir(summary_path))
    return project_path


def summary(parser):
    import os
    import pandas as pd

    summary_path = project_path + parser.get('folders', 'summary_path')
    summary_file = parser.get('files', 'summary')
    os.chdir(summary_path)
    # print(os.listdir(summary_path))
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
    session_id = cluster.Key + '-' + cluster.BirdID + '-' + cluster.TaskName + '-' + cluster.TaskSession + '-' + cluster.SessionDate + '-Site' + cluster.Site
    cell_id = session_id + '-' + cluster.Channel + '-' + cluster.Cluster
    session_path = project_path + '\\' + cluster.BirdID + '\\' + cluster.TaskName + '\\' + cluster.TaskSession + '(' + cluster.SessionDate + ')'
    cell_path = session_path + '\\' + cluster.Site + '\\Songs'
    # print('Accessing... ' + cell_path)
    # os.chdir(cell_path)
    return session_id, cell_id, session_path, cell_path




if __name__ == '__main__':
    config_file: str = 'project.ini'
    parser = config(config_file)
    project_path = project(parser)
    summary_cluster, nb_cluster = summary(parser)
