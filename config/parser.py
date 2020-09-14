import os
import toml

class ConfigParser():

    def __init__(self, config_file):

        self._file = os.path.join(os.path.split(__file__)[0], config_file)

    def load(self):

        return toml.load(self._file)