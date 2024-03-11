from flask import Flask, jsonify, request
# Import both handler functions
from api.summarize import scrape_and_summarize
from api.vectorsearch import search
from api.nl2sql import askquestion


from openai import AzureOpenAI
import os

api_base =  os.environ.get('API_BASE') # "https://ai-dictate.openai.azure.com/"  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
api_version = os.environ.get('API_VERSION') # '2023-05-15'  # this might change in the future
deployment_name = os.environ.get('DEPLOYMENT_NAME')

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=api_base,
)

app = Flask(__name__)

@app.route('/summarize', methods=['GET'])
def summarize():
    return scrape_and_summarize(request, client, deployment_name)
    
@app.route('/vectorsearch', methods=['POST'])
def vectorsearch():
    return search(request)    
    
@app.route('/ask_question', methods=['POST'])
def ask():    
    return askquestion(request, client)  
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)    