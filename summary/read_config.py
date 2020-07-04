from configparser import ConfigParser

global parser

config_file = 'summary/project.ini'
parser = ConfigParser()
parser.read(config_file)
print(parser.sections())
print('JR')

#TODO: needs to be fixed, doesn't work because of the scope