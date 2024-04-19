import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import os

# Check if GPU is available and select the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Load the model to the specified device
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

def get_embeddings(texts):
    try:
        # Encode the inputs using the tokenizer
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move output tensors back to CPU for numpy conversion
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

files = []
embeddings = []

directory_path = "holmes"



for filename in os.listdir(directory_path):
    print(filename)
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()    
        embedding = get_embeddings(content).tolist()
        files.append(filename)
        embeddings.append(embedding[0])
        print (embedding[0])

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