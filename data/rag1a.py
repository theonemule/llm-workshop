import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import os

# Check if GPU is available and select the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_embeddings(texts):
    try:
        # Encode the inputs using the tokenizer
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Return embeddings for each chunk
    except Exception as e:
        print("Error occurred while generating embeddings:", e)
        return None

# Connect to Milvus and define the collection schema with metadata fields
connections.connect("default", host='localhost', port='19530')

collection_name = "sherlock"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=64),
]
schema = CollectionSchema(fields, description="Document Embeddings with Metadata")
collection = Collection(name=collection_name, schema=schema)

# Prepare to store each chunk's embedding separately
chunk_files = []  # Filenames for each chunk
chunk_embeddings = []  # Embeddings for each chunk

directory_path = "holmes"

for filename in os.listdir(directory_path):
    print(filename)
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            chunk_index = 0  # Keep track of chunk index for a given file
            while True:
                content_chunk = file.read(510)
                if not content_chunk:
                    break  # End of file
                
                embedding = get_embeddings(content_chunk)
                if embedding is not None:
                    # Store each chunk's embedding separately along with a modified filename
                    chunk_filename = f"{filename}_chunk_{chunk_index}"
                    chunk_files.append(chunk_filename)
                    chunk_embeddings.append(embedding[0].tolist())
                    chunk_index += 1

entities = [chunk_embeddings, chunk_files]
    
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
print("Chunks and embeddings with metadata have been successfully ingested into Milvus.")
