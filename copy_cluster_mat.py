from summary import load

config_file = 'project.ini'
parser = load.config(config_file)
projectROOT = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file
del config_file, parser

def copy_cluster_mat(summary_cluster, projectROOT):

    # for cluster_run in range(0, nb_cluster):
    cluster_run = 0
    cluster = load.cluster(summary_cluster, cluster_run)
    print(cluster)

# pass

    return cluster










if __name__ == '__main__':
    cluster = copy_cluster_mat(summary_cluster, projectROOT)