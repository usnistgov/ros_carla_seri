
'''
Common functions to all modules

Returns:
    dict: dictionary with the yaml file content or empty dict if the file is not found
'''

import sys
import yaml


def read_yaml(path):
    '''
    Method to read a yaml file

    Args:
        path (str): absolute path to the yaml file
    '''

    with open(path, "r", encoding="utf8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError:
            sys.stderr.write("Unable to read configuration file")
            return {}
