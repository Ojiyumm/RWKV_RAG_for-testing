from src.services.files_service import chunk_file_contents
from src.services.files_service import FileStatusManager, FileStatusMonitorThread


from threading import Thread

file_name = '/home/rwkv/Peter/RaqQ-master/README.md'
file_content = chunk_file_contents(file_name)
db_path = '/home/rwkv/Peter/RaqQ-master/src/services/chroma/files_services.db'
fsm = FileStatusManager(db_path)
fsm.set_file_status(file_name,'processing')
print(fsm.add_file_chunks(file_name,file_content))
print(fsm.get_file_status(file_name))
print(fsm.get_chunks_status(file_name))
fsm.set_chunk_status(file_name,0,'processed')
print(fsm.get_chunk_content(file_name,0))
print(fsm.get_chunk_uuid(file_name,0))
print(fsm.get_chunks_status(file_name))
fsm.delete_file(file_name)
print(fsm.get_file_status(file_name))
fsm.close()
monitor_thread = FileStatusMonitorThread(fsm)
t = Thread(target=monitor_thread.run)
t.start()
t.join()