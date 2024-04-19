import pandas as pd
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import numpy as np  # Import numpy for averaging
import tiktoken
import os
from openai import AzureOpenAI

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
# Load the model to the specified device

client = AzureOpenAI(
    api_key = "e34a2369778a4c0ebee164e8fceca658", # os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2023-05-15",
    azure_endpoint = "https://ai-dictate.openai.azure.com/" # os.getenv("AZURE_OPENAI_ENDPOINT")
)


def get_embeddings(texts):
    try:
        # Encode the inputs using the tokenizer
        inputs = tokenizer.encode(texts)        
        return client.embeddings.create(input = [texts], model="rag-demo").data[0].embedding
    except Exception as e:
        print("Error occurred while generating embeddings:", e)
        return None
        
def get_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
        

# Connect to Milvus and define the collection schema with metadata fields
connections.connect("default", host='localhost', port='19530')

collection_name = "sherlock"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=64),
]
schema = CollectionSchema(fields, description="Document Embeddings with Metadata")
collection = Collection(name=collection_name, schema=schema)

directory_path = "holmes"
files = []
embeddings = []

for filename in os.listdir(directory_path):
    print(filename)
    document_embeddings = []  # Store embeddings for the current document
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read(510)

        tokens = tokenizer.encode(content)
        
        for chunk in get_chunks(tokens, 8192):
            text = tokenizer.decode(chunk)
            
            embedding = get_embeddings(text)
            
            files.append(filename)
            embeddings.append(embedding)
        
                

# Define entities for Milvus insertion
entities = [
   embeddings,
   files
]

# Insert data into Milvus
insert_response = collection.insert(entities)

# Define index parameters for the "embedding" field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 100},
}

# Create the index
collection.create_index(field_name="embedding", index_params=index_params)
print("Index on the 'embedding' field has been created.")
print("Documents and embeddings with metadata have been successfully ingested into Milvus.")
