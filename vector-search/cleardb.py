from pymilvus import connections, utility

# Connect to the Milvus server
connections.connect("default", host='localhost', port='19530')

# List all collections
collections = utility.list_collections()

# Drop each collection
for collection_name in collections:
    print(f"Dropping collection: {collection_name}")
    utility.drop_collection(collection_name)

print("All collections have been dropped.")