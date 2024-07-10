#coding:utf-8
import os
from src.services.helpers import ServiceWorker as _ServiceWorker
from src.tools.make_data import Jsonl2Binidx


class ServiceWorker(_ServiceWorker):
    """
    TODO 待定，不确定参数是前端web输入还是放在配置文件里
    """
    def init_with_config(self, config):
        self.jsonl_path = config.get('jsonl_path')
        self.output_path = config.get('output_path')
        self.n_epoch = config.get('n_epoch')
        self.context_len = 1024

    def process(self, cmd):
        if cmd['cmd'] == "MAKE_DATA":
            # 优先选取用户自定义输入数据，否则去配置文件获取默认值
            jsonl_file = cmd['jsonl_file'] or self.jsonl_path
            context_len = cmd['context_len'] or self.context_len
            output_path = cmd['output_path'] or self.output_path
            n_epoch = cmd['n_epoch'] or self.n_epoch
            is_str = cmd['is_str']

            if not is_str and not os.path.exists(jsonl_file):
                raise FileNotFoundError(f"{jsonl_file} not found")
            if not os.path.exists(output_path):
                raise NotADirectoryError(f"{output_path} not found")

            obj = Jsonl2Binidx(jsonl_file=jsonl_file, n_epoch=n_epoch,
                         output_path=output_path, context_len=context_len,
                         is_str=is_str)
            obj.run()
            return obj.output_file_name_prefix
        else:
            return ServiceWorker.UNSUPPORTED_COMMAND

