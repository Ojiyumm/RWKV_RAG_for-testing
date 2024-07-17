#coding=utf-8
import msgpack
import zmq


class RWKVPEFTClient(object):
    def __init__(self,frontend_url: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ) # 设置请求套接字
        self.socket.connect(frontend_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000) # TODO 设置接受操作超时时间，是否合理，如果文件很大，转换时间会很长


    def lora_train(self,**kwargs):

        kwargs.update({"cmd":"LORA"})
        self.socket.send(msgpack.packb(kwargs, use_bin_type=True))
        response = self.socket.recv()
        return msgpack.unpackb(response, raw=False)

    def state_tuning_train(self,**kwargs):

        kwargs.update({"cmd":"STATE_TUNING"})
        self.socket.send(msgpack.packb(kwargs, use_bin_type=True))
        response = self.socket.recv()
        return msgpack.unpackb(response, raw=False)

    def pissa_train(self,**kwargs):
        kwargs.update({"cmd":"PISSA"})
        self.socket.send(msgpack.packb(kwargs, use_bin_type=True))
        response = self.socket.recv()
        return msgpack.unpackb(response, raw=False)