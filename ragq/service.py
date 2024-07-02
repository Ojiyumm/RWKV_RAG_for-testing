import sys
import os
import argparse
import importlib
from configuration import ragq_base_dir
from configuration import Configuration
 # 添加其它依赖项目
parent_dir = os.path.dirname(ragq_base_dir)  #ragq的上一级
ext_project_package_dir = os.path.join(parent_dir, 'third_party_packages')
required_projects = ['rwkv_lm_ext', 'rwkv_peft', 'fla', 'cuda']  # 需要依赖的第三方项目包

for project_name in required_projects:
    ext_project_dir = os.path.join(ext_project_package_dir, project_name)
    if not os.path.exists(ext_project_dir):
        raise NotADirectoryError(f"This project requires the  {project_name} Please place it in the path {ext_project_dir}.")
    # 检查有没有加__init__.py文件
    if project_name == 'cuda':
        continue
    if not os.path.exists(os.path.join(ext_project_dir, '__init__.py')):
        raise FileNotFoundError(f"third-party project {project_name} must have an __init__.py file in the project directory.")


sys.path.insert(0, ext_project_package_dir) # 加入第三方包的路径

from src.services.helpers import start_service


if __name__ == "__main__":
    service_package_base = 'src.services.'
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
            service_module_name = service_package_base + service_module_name
            is_init_once = config_service.get("is_init_once",False)
            if is_init_once:
                print(f"Init once for {service_module_name}")
                module = importlib.import_module(service_module_name)
                module.init_once(config_service)
            start_service(service_module_name,config_service)