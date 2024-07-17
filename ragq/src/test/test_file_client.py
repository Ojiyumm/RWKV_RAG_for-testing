from src.clients.file_client import FileClient



front_end_url = "tcp://localhost:7785"
file_client = FileClient(front_end_url)
file_name = "/home/yueyulin/tmp/1.txt"
print(file_client.check_file_exists(file_name))
print(file_client.add_file(file_name))
print(file_client.check_file_exists(file_name))