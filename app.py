# pip install flask openai whisper werkzeug BeautifulSoup4 tiktoken pymilvus tensor pandas milvus transformers tensor openai-whisper, ffmpeg

# DEPLOYMENT_NAME=ai-dictate
# AZURE_OPENAI_ENDPOINT=https://ai-dictate.openai.azure.com/
# OPENAI_API_VERSION=2023-05-15
# AZURE_OPENAI_API_KEY=e34a2369778a4c0ebee164e8fceca658

from flask import Flask, request, jsonify, send_from_directory
from api.summarize import scrape_and_summarize
from api.vectorsearch import search
from api.nl2sql import askquestion
from api.dictate import transcribe_audio
from api.quote import generate_quote


from openai import AzureOpenAI
import os

api_base =  os.environ.get('API_BASE') # "https://ai-dictate.openai.azure.com/"  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
api_version = os.environ.get('API_VERSION') # '2023-05-15'  # this might change in the future
deployment_name = os.environ.get('DEPLOYMENT_NAME') # ai-dictate

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=api_base,
)

app = Flask(__name__, static_folder='static')

@app.route('/summarize', methods=['GET'])
def summarize():
    return scrape_and_summarize(request, client, deployment_name)

@app.route('/quote', methods=['POST'])
def quote():
    return generate_quote(request, client, deployment_name)    
  
@app.route('/vectorsearch', methods=['POST'])
def vectorsearch():
    return search(request)    
    
@app.route('/ask_question', methods=['POST'])
def ask():    
    return askquestion(request, client, deployment_name)  

@app.route('/dictate', methods=['POST'])
def dictate():    
    return transcribe_audio(request, client, deployment_name)  

    
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')    
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)    