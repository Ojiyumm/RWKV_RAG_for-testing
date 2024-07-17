#coding:utf-8

from helpers import ServiceWorker as _ServiceWorker
from tools.make_data import Jsonl2Binidx


class ServiceWorker(_ServiceWorker):
    """
    TODO 待定，不确定参数是前端web输入还是放在配置文件里
    """
    pass
    
    def init_with_config(self, config):
        pass
    
    def process(self, cmd):
        if cmd['cmd'] == "MAKE_DATA":
            jsonl_file = cmd['jsonl_file']
            context_len = cmd['context_len']
            output_path = cmd['output_path']
            n_epoch = cmd['n_epoch']
            Jsonl2Binidx(jsonl_file=jsonl_file, n_epoch=n_epoch,
                         output_path=output_path, context_len=context_len).run()
        else:
            return ServiceWorker.UNSUPPORTED_COMMAND

