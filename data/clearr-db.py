from pymilvus import Collection, utility, connections
import os

# Step 1: Connect to the Milvus server
# Replace 'default' with your connection name if different.
# Adjust the host and port accordingly if your Milvus instance is not running with the default settings.

name = "document_embeddings_with_metadata"

milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = int(os.getenv('MILVUS_PORT', '19530'))
connections.connect("default", host=milvus_host, port=milvus_port)

utility.drop_collection(name)

connections.disconnect("default")

print(name + " has been deleted.")