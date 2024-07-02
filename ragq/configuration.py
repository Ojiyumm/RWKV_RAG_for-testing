import yaml
import os

class Configuration:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)


# ragq 向量路径
ragq_base_dir = os.path.dirname(os.path.abspath(__file__))