#coding=utf-8
import msgpack
import zmq


class Jsonl2BinIdxClient(object):
    def __init__(self,frontend_url: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ) # 设置请求套接字
        self.socket.connect(frontend_url) # TODO 不设置上下文管理协议正确关闭连接吗？？？
        self.socket.setsockopt(zmq.RCVTIMEO, 60000) # TODO 设置接受操作超时时间，是否合理，如果文件很大，转换时间会很长


    def transform(self,jsonl_file: str=None,n_epoch: int=3,output_path: str=None,context_len: int=1024, is_str=False):
        """
        jsonl转换成binidx
        """
        cmd = {"cmd": "MAKE_DATA", 'jsonl_file': jsonl_file, 'n_epoch': n_epoch,
               'output_path': output_path, 'context_len': context_len, 'is_str': is_str}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        response = self.socket.recv()
        return msgpack.unpackb(response, raw=False)

