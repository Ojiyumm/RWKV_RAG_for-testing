#coding=utf-8
import msgpack
import zmq


class JsonL2BinIdxClient(object):
    def __init__(self,frontend_url: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ) # 设置请求套接字
        self.socket.connect(frontend_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000) # TODO 设置接受操作超时时间，是否合理，如果文件很大，转换时间会很长


    def transform(self,jsonl_file: str,n_epoch: int,output_path: str,context_len: int):
        """
        jsonl转换成binidx
        """
        cmd = {"cmd": "MAKE_DATA", 'jsonl_file': jsonl_file, 'n_epoch': n_epoch,
               'output_path': output_path, 'context_len': context_len}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        response = self.socket.recv()
        return msgpack.unpackb(response, raw=False)
    
if __name__ == '__main__':
    client = JsonL2BinIdxClient('tcp://localhost:7787')
    client.transform('/home/rwkv/Peter/Data/Telechat5/top_1k.jsonl',3,
                     '/home/rwkv/Peter/Data/Telechat5',1024)
    