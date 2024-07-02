from src.clients.jsonl2binidx_client import Jsonl2BinIdxClient


client = Jsonl2BinIdxClient('tcp://localhost:7787')
client.transform('/home/rwkv/Peter/Data/Telechat5/top_2k.jsonl',3,
                     '/home/rwkv/Peter/Data/Telechat5',1024)