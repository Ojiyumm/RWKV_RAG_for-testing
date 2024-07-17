from src.clients.cache_client import CacheClient


cache_client = CacheClient("tcp://localhost:7777","tcp://localhost:7779")
print(cache_client.exists("key1"))
print(cache_client.exists("key2"))
print(cache_client.get("key1"))
print(cache_client.get("key2"))
print(cache_client.set("key1", "value1"))
print(cache_client.get("key1"))
print(cache_client.exists("key1"))
print(cache_client.delete("key1"))
print(cache_client.exists("key1"))
print(cache_client.get("key1"))

print(cache_client.set("key2", {"name": "John Doe", "age": 30,"codes":[1,2,3],"embeddings":[1.0,2.1111,-0.3323]}))
print(cache_client.exists("key2"))
print(cache_client.get("key2"))
print(cache_client.delete("key2"))
print(cache_client.exists("key2"))