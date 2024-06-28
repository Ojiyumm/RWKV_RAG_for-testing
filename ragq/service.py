import sys
import os
import argparse
import importlib
sys.path.insert(0, os.path.abspath('../third_party_packages')) # 加入第三方包的路径

from configuration import Configuration
from src.services.helpers import start_service


if __name__ == "__main__":
    package_base = 'src.services.'
    parser = argparse.ArgumentParser(description="Service start")
    parser.add_argument("--service_config", help="Service configuration file",type=str,default="ragq.yml")
    args = parser.parse_args()
    print(args.service_config)
    config = Configuration(args.service_config)
    services = config.config.keys()
    print(f"Starting services {services}")
    for service in services:
        if config.config[service]["enabled"]:
            config_service = config.config[service]
            print(f"Starting service {service}")
            service_module_name = config_service["service_module"]
            service_module_name = package_base + service_module_name
            is_init_once = config_service.get("is_init_once",False)
            if is_init_once:
                print('ffffffffffffffffffffffffffffffffffffffffffffffffffff')
                print(f"Init once for {service_module_name}")
                module = importlib.import_module(service_module_name)
                module.init_once(config_service)
            start_service(service_module_name,config_service)