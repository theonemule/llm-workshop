import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import os

# Check if GPU is available and select the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model to the specified device
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

def get_chunk_embeddings(text):
    # Split the text into chunks of 510 tokens to leave space for [CLS] and [SEP] tokens
    max_chunk_size = 510
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]
    
    chunk_embeddings = []
    
    i = 0
    
    for chunk in chunks:
        i = i + 1
        print(f'{i} of {len(chunks)}')
        # Encode the inputs using the tokenizer
        inputs = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Take the mean of the last hidden state to get a single embedding vector per chunk
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        chunk_embeddings.append(chunk_embedding.squeeze().tolist())  # Convert to list and remove extra dimensions
    
    return chunk_embeddings

# Connect to Milvus
connections.connect("default", host='localhost', port='19530')

# Define or ensure the collection exists as shown previously

collection_name = "sherlock_chunks"
if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Document Chunk Embeddings with Metadata")
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

directory_path = "holmes"

for filename in os.listdir(directory_path):
    print(filename)
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()    
        chunk_embeddings = get_chunk_embeddings(content)
        
        for i, embedding in enumerate(chunk_embeddings):
            print(i)
            entities = {
                "embedding": [embedding],
                "filename": [filename],
                "chunk_index": [i],
            }
            # Insert data into Milvus for each chunk
            insert_response = collection.insert(entities)

# Define and create the index for the "embedding" field as shown previously
