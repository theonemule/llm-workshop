import openai
from transformers import GPT2Tokenizer
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
import re
import os

MAX_TOKENS=8191
openai.api_type = "azure"
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')

def read_and_chunk_novel(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    paragraphs = [p for p in text.split('\n') if p]
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        tokens = tokenizer(paragraph)['input_ids']
        current_chunk_token_count = sum(len(tokenizer(chunk)['input_ids']) for chunk in current_chunk)

        if current_chunk_token_count + len(tokens) > MAX_TOKENS:
            chunks.append("\n".join(current_chunk))
            current_chunk = [paragraph]
        else:
            current_chunk.append(paragraph)
    chunks.append("\n".join(current_chunk))  # Add the last chunk
    return "\n\n".join(chunks)


text = read_and_chunk_novel("gatsby.txt")

response = openai.Embedding.create(
    input=text
    engine="rag-demo"
)
embeddings = response['data'][0]['embedding']
print(len(embeddings))