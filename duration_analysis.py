from summary.read_config import parser
from summary import load

project_path = load.project(parser)  # find the project folder
summary_cluster, nb_cluster = load.summary(parser)  # load cluster summary file

data_path = project_path + '\Analysis\SAP_features'
analysis_file = 'SAP(ALL).txt'


