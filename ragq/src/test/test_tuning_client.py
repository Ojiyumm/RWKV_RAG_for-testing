from src.clients.tuning_client import RWKVPEFTClient


client = RWKVPEFTClient('tcp://localhost:7789')
client.state_tuning_train(load_model='/home/rwkv/Peter/model/base/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
                          proj_dir='/home/rwkv/LongHua/project_log',
                          data_file='/home/rwkv/Peter/Data/Telechat5/top_2k',
                          micro_bsz=2)
