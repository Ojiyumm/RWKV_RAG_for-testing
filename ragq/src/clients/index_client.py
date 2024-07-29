import uuid

import msgpack
import zmq


class IndexClient:
    def __init__(self,frontend_url) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(frontend_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 60000)

    def index_texts(self,texts,keys=None,collection_name=None):
        if keys is None or isinstance(keys, list) == False or len(keys) != len(texts):
            keys = [str(uuid.uuid4()) for i in range(len(texts))]
        cmd = {"cmd": "INDEX_TEXTS", "texts": texts,"keys": keys,'collection_name':collection_name}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        resp["keys"] = keys
        return resp

    def show_collection(self):
        cmd = {"cmd":'SHOW_COLLECTIONS'}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp

    def create_collection(self,collection_name=None):
        cmd = {"cmd":'CREATE_COLLECTION','collection_name':collection_name}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp

    def delete_collection(self,collection_name=None):
        cmd = {"cmd":'DELETE_COLLECTION','collection_name':collection_name}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp

    def search_nearby(self,text,collection_name):
        cmd = {"cmd": "SEARCH_NEARBY", "text": text, 'collection_name':collection_name}
        self.socket.send(msgpack.packb(cmd, use_bin_type=True))
        msg = self.socket.recv()
        resp = msgpack.unpackb(msg, raw=False)
        return resp
