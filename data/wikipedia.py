import rdflib
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Check if GPU is available and select the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Load the model to the specified device
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)


def text_to_vector(text):
    try:
        # Encode the inputs using the tokenizer
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move output tensors back to CPU for numpy conversion
    except Exception as e:
        print("Error occurred while generating embeddings:", e)
        return None
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

collection_name = 'minipedia'
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
]
schema = CollectionSchema(fields, description="Document Embeddings with Metadata")


collection = Collection(name=collection_name, schema=schema)  # Assuming collection is already created

# Load the TTL file
g = rdflib.Graph()
g.parse("long-abstracts_lang.ttl", format="turtle")

embeddings = []
texts = []
batch_size = 1000
count = 0

for subj, pred, obj in g:
    vector = text_to_vector(str(obj))
    original_text = str(obj)
    
    embeddings.append(vector)
    texts.append(original_text)
    
    if len(embeddings) >= batch_size:
        print(count)
        entities = [
            embeddings,
            texts
        ]
        mr = collection.insert(entities)
        # print(f"Inserted a batch into Milvus with IDs: {mr.primary_keys}")
        embeddings = []
        texts = []
        count = count+1

if embeddings:
    entities = [
        embedding,
        texts
    ]
    mr = collection.insert(entities)
    print(f"Inserted the final batch into Milvus with IDs: {mr.primary_keys}")
    
# Create the index
# collection.create_index(field_name="embedding", index_params=index_params)
print("Index on the 'embedding' field has been created.")    

connections.disconnect("default")