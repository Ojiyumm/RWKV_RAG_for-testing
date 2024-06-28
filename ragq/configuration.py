import yaml

class Configuration:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
