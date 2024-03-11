import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

# Initialize tokenizer and model for embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(texts):
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    except Exception as e:
        print("Error occurred while generating embeddings:", e)
        return None

# Connect to Milvus and define the collection schema with metadata fields
connections.connect("default", host='localhost', port='19530')

collection_name = "document_embeddings_with_metadata"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="ShowNumber", dtype=DataType.INT64),
    FieldSchema(name="AirDate", dtype=DataType.VARCHAR, max_length=64),  # Assuming date as string; adjust if using a timestamp
    FieldSchema(name="Round", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="Category", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="Value", dtype=DataType.FLOAT),
    FieldSchema(name="Question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="Answer", dtype=DataType.VARCHAR, max_length=1024),
]
schema = CollectionSchema(fields, description="Document Embeddings with Metadata")
collection = Collection(name=collection_name, schema=schema)

# Define batch size
batch_size = 100
count = 0

# Load and process data from CSV in batches
for batch_df in pd.read_csv('JEOPARDY.csv', chunksize=batch_size):

    print(count)
    count = count + 1
    
    # Combine title and content (or any fields you wish) for embedding
    batch_df['text_for_embedding'] = batch_df['Category'] + " " + batch_df['Question'] + " " + batch_df['Answer']



    # Generate embeddings for each document in the batch
    batch_df['embeddings'] = batch_df['text_for_embedding'].apply(lambda x: get_embeddings([x])[0] if get_embeddings([x]) is not None else None).tolist()
    batch_df['Value'] = batch_df['Value'].apply(lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) else x)

    # Remove rows with None embeddings
    batch_df = batch_df[batch_df['embeddings'].notna()]
    
    # Extract entities for insertion
    entities = [
        batch_df['embeddings'].tolist(),  # embedding field
        batch_df['ShowNumber'].tolist(),  # url field
        batch_df['AirDate'].tolist(),  # url field
        batch_df['Round'].tolist(),  # url field
        batch_df['Category'].tolist(),  # url field
        batch_df['Value'].tolist(),  # url field
        batch_df['Question'].tolist(),  # url field
        batch_df['Answer'].tolist(),  # url field
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
