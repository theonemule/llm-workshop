from flask import Flask, request, jsonify
from pymilvus import Collection, connections
from transformers import BertTokenizer, BertModel
import torch
import numpy as np  # For converting tensors to numpy arrays
import tiktoken
from openai import AzureOpenAI

app = Flask(__name__)

# Connect to Milvus
connections.connect("default", host='localhost', port='19530')

# Specify the collection name and ensure it's loaded
collection_name = "sherlock"
collection = Collection(name=collection_name)
collection.load()

# Load the model and tokenizer for BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

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
        return client.embeddings.create(input = [texts], model="rag-demo").data[0].embedding
    except Exception as e:
        print("Error occurred while generating embeddings:", e)
        return None

# def get_embeddings(texts):
    # try:
        # # Encode the inputs using the tokenizer
        # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device
        # with torch.no_grad():
            # outputs = model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move output tensors back to CPU for numpy conversion
    # except Exception as e:
        # print("Error occurred while generating embeddings:", e)
        # return None

@app.route('/search', methods=['POST'])
def search():
    # Extract query from request
    data = request.json
    query_text = data.get('query')
    if not query_text:
        return jsonify({"error": "Please provide a 'query' field."}), 400
    
    # Generate query embedding
    query_embedding = get_embeddings(query_text)
    # query_embeddings_list = query_embedding.tolist()
    
   
    # Search parameters
    # search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    search_params={
        "metric_type": "L2"
    }
    
    # Perform the search
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["id", "embedding", "filename"]
    )
    
    # Process and return search results
    search_results = []
    for hits in results:
        for hit in hits:
            search_results.append({
                "id": hit.id,
                "distance": hit.distance,
                "filename": hit.entity.get('filename')
            })
    
    return jsonify(search_results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
