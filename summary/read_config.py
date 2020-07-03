from configparser import ConfigParser

config_file = '../project.ini'
parser = ConfigParser()
parser.read(config_file)
print(parser.sections())


#TODO: needs to be fixed, doesn't work because of the scope