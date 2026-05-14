# LLM Workshop - Multi-Demo Application

A comprehensive demonstration suite showcasing various LLM capabilities including RAG (Retrieval-Augmented Generation), vector search, NL2SQL, audio transcription, text summarization, and more.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Demo Descriptions](#demo-descriptions)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

## Overview

This LLM Workshop is a multi-demo Flask application that showcases different AI/ML capabilities powered by Azure OpenAI and Milvus vector database. It includes multiple demonstration modules that can be run independently or together.

**Key Technologies:**
- **Framework**: Flask (Python Web Framework)
- **LLM**: Azure OpenAI (GPT models)
- **Vector Database**: Milvus (for semantic search and RAG)
- **Embeddings**: BERT, DistilBERT, and Azure OpenAI embeddings
- **Audio Processing**: OpenAI Whisper

## Features

### Core Capabilities

1. **Vector Search** - Semantic search across document embeddings
2. **RAG (Retrieval-Augmented Generation)** - Retrieve relevant documents and generate context-aware responses
3. **NL2SQL** - Convert natural language questions to SQL queries
4. **Audio Transcription** - Convert speech to text using Whisper
5. **Text Summarization** - Generate summaries of web content
6. **Quote Generation** - Generate AI-powered quotes
7. **Text Tokenization** - Count and analyze tokens in text
8. **Resume Search** - Search and analyze resume data with BERT embeddings

### Multiple Implementation Approaches

The repository includes various implementations demonstrating different approaches:
- **rag.py** - Original RAG implementation
- **rag1.py** - Enhanced RAG with BERT embeddings
- **rag3.py** - Sherlock text collection implementation
- **rag4.py** - Advanced RAG with OpenAI embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Flask Web Application                      │
├─────────────────────────────────────────────────────────────┤
│                      API Layer (app.py)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  RAG Module  │  │ NL2SQL Query │  │Vector Search │      │
│  │  (ragapi.py) │  │  (nl2sql.py) │  │(vectorsearch)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Transcriber  │  │ Summarizer   │  │Quote Gen     │      │
│  │  (dictate)   │  │(summarize.py)│  │(quote.py)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                        │                   │
         ▼                        ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Azure OpenAI     │  │  Milvus Vector   │  │  SQLite Database │
│ (Embeddings &    │  │    Database      │  │  (QA Data)       │
│  Completions)    │  │  (Vector Search) │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **CUDA** (optional): For GPU acceleration with Torch
- **Docker** (optional): For containerized deployment

### Required Services

1. **Milvus Vector Database**: 
   - Docker: `docker run -d --name milvus-server -p 19530:19530 milvus/milvus:latest`
   - Or local installation from https://milvus.io/docs/install_standalone-docker.md

2. **Azure OpenAI Account**:
   - Azure subscription with OpenAI service enabled
   - API key and endpoint from Azure Portal

### Python Dependencies

```bash
Flask                    # Web framework
openai                   # Azure OpenAI SDK
whisper                  # Audio transcription
BeautifulSoup4          # Web scraping
tiktoken                # Token counting
pymilvus                # Milvus client
transformers            # BERT/DistilBERT models
torch                   # PyTorch for embeddings
pandas                  # Data processing
```

## Installation & Setup

### 1. Clone or Download Repository

```bash
cd llm-workshop
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask openai whisper werkzeug BeautifulSoup4 tiktoken pymilvus pandas milvus transformers torch openai-whisper
```

### 4. Start Milvus Vector Database

**Using Docker:**
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

**Verify connection:**
```bash
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port=19530); print('Connected to Milvus')"
```

## Configuration

### 1. Environment Variables

Copy `.env.example` to `.env` and update with your values:

```bash
cp .env.example .env
```

### 2. Edit `.env` File

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_actual_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Flask Configuration
FLASK_DEBUG=True
FLASK_ENV=development
```

**Important Security Notes:**
- ⚠️ Never commit `.env` file to version control
- Use strong, unique API keys
- Rotate keys regularly
- Use different keys for development and production
- Restrict API key permissions in Azure Portal

### 3. Initialize Data (Optional)

Initialize Milvus collections and import data:

```bash
# Import Jeopardy data with embeddings
python data/import-milvus.py

# Import resume data
python data/import-resumes-bert.py

# Import Sherlock text data
python data/rag3.py
```

## Running the Application

### Development Mode

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Mode with Docker

1. **Build Docker image:**
```bash
docker build -t llm-workshop:latest .
```

2. **Run container:**
```bash
docker run -p 5000:5000 \
  -e AZURE_OPENAI_API_KEY=your_key \
  -e AZURE_OPENAI_ENDPOINT=your_endpoint \
  -e MILVUS_HOST=milvus \
  --network llm-workshop-network \
  llm-workshop:latest
```

3. **Docker Compose (Recommended):**

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      COMMON_STORAGETYPE: local
    volumes:
      - milvus-data:/var/lib/milvus

  app:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - milvus
    environment:
      AZURE_OPENAI_API_KEY: ${AZURE_OPENAI_API_KEY}
      AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT}
      MILVUS_HOST: milvus
      MILVUS_PORT: 19530
    volumes:
      - ./data:/app/data

volumes:
  milvus-data:
```

Run with Docker Compose:
```bash
docker-compose up
```

## API Endpoints

### Vector Search
**POST** `/vectorsearch`
```json
{
  "query": "machine learning"
}
```

### RAG Search
**POST** `/rag`
```json
{
  "query": "What is artificial intelligence?"
}
```

### NL2SQL Query
**POST** `/ask_question`
```json
{
  "question": "Show all Jeopardy answers in the Science category"
}
```

### Audio Transcription
**POST** `/dictate`
```
multipart/form-data
audio_file: [binary audio data]
```

### Text Summarization
**GET** `/summarize?url=https://example.com`

### Quote Generation
**POST** `/quote`
```json
{
  "topic": "artificial intelligence"
}
```

### Token Count
**POST** `/tokenize`
```json
{
  "text": "Your text to tokenize"
}
```

### Resume Search
**GET** `/resume`

### Home Page
**GET** `/`

## Demo Descriptions

### 1. Vector Search Demo
Searches across embedded documents using semantic similarity.
- **File**: `api/vectorsearch.py`
- **Model**: DistilBERT
- **Use Case**: Find relevant documents based on meaning, not just keywords

### 2. RAG (Retrieval-Augmented Generation)
Combines document retrieval with generative AI for context-aware responses.
- **File**: `api/ragapi.py`
- **Model**: BERT embeddings + GPT-4
- **Use Case**: Answer questions based on specific documents/domain knowledge

### 3. NL2SQL Query Generator
Converts natural language questions to SQL queries.
- **File**: `api/nl2sql.py`
- **Dataset**: Jeopardy Q&A database
- **Use Case**: Query databases using plain English

### 4. Audio Transcription
Transcribes audio files to text using OpenAI Whisper.
- **File**: `api/dictate.py`
- **Model**: Whisper (all languages)
- **Use Case**: Convert speech to text for accessibility or data input

### 5. Web Summarization
Scrapes and summarizes web page content.
- **File**: `api/summarize.py`
- **Use Case**: Quickly understand web content without reading entire pages

### 6. Quote Generation
Generates topic-specific quotes using GPT.
- **File**: `api/quote.py`
- **Use Case**: Creative content generation

### 7. Token Counting
Analyzes token usage for LLM cost estimation.
- **File**: `api/tokenization.py`
- **Use Case**: Estimate API costs before processing large texts

## Development

### Project Structure

```
llm-workshop/
├── app.py                      # Main Flask application
├── api/                        # API endpoint implementations
│   ├── ragapi.py              # RAG search endpoints
│   ├── nl2sql.py              # SQL generation
│   ├── vectorsearch.py        # Vector search
│   ├── dictate.py             # Audio transcription
│   ├── summarize.py           # Web summarization
│   ├── quote.py               # Quote generation
│   ├── tokenization.py        # Token counting
│   ├── rag.py, rag1.py        # Alternative RAG implementations
│   └── ragapi.py              # Enhanced RAG with BERT
├── data/                       # Data processing and import scripts
│   ├── import-milvus.py       # Import data to Milvus
│   ├── import-milvus-bert.py  # BERT embeddings import
│   ├── import-resumes-bert.py # Resume data import
│   ├── import-sqlite.py       # SQLite database import
│   ├── JEOPARDY.csv           # Sample Q&A dataset
│   ├── rag*.py                # Various RAG implementations
│   └── docs/                  # Sample documents by category
├── static/                     # Frontend files
│   ├── index.html             # Web UI
│   └── images/                # UI assets
├── .env.example               # Environment variables template
├── dockerfile                 # Docker configuration
├── docker-compose.yml         # Multi-container setup
├── start.sh                   # Startup script
└── requirements.txt           # Python dependencies
```

### Adding New Features

1. Create new endpoint in `api/` directory
2. Import in `app.py`
3. Add route with `@app.route()`
4. Update documentation

### Testing Endpoints Locally

Using curl:
```bash
# Vector search
curl -X POST http://localhost:5000/vectorsearch \
  -H "Content-Type: application/json" \
  -d '{"query":"your search term"}'

# NL2SQL
curl -X POST http://localhost:5000/ask_question \
  -H "Content-Type: application/json" \
  -d '{"question":"SELECT * FROM QuestionsAnswers"}'
```

Using Python:
```python
import requests

# Vector search
response = requests.post('http://localhost:5000/vectorsearch', 
    json={'query': 'machine learning'})
print(response.json())

# RAG
response = requests.post('http://localhost:5000/rag',
    json={'query': 'What is AI?'})
print(response.json())
```

## Troubleshooting

### Issue: "Cannot connect to Milvus"

**Solution:**
1. Verify Milvus is running: `docker ps | grep milvus`
2. Check connection parameters in `.env`
3. Restart Milvus:
   ```bash
   docker restart milvus-standalone
   ```

### Issue: "Azure OpenAI API Key Invalid"

**Solution:**
1. Verify key in Azure Portal
2. Check `.env` file has correct values
3. Ensure key hasn't expired
4. Test with: 
   ```bash
   python -c "from openai import AzureOpenAI; import os; client = AzureOpenAI(api_key=os.getenv('AZURE_OPENAI_API_KEY'), azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), api_version='2023-05-15')"
   ```

### Issue: "Out of memory when loading models"

**Solution:**
1. Reduce batch size in import scripts
2. Use smaller model versions (DistilBERT vs BERT)
3. Enable GPU: Install CUDA and PyTorch with GPU support
4. Process data in chunks rather than all at once

### Issue: "Module not found errors"

**Solution:**
1. Activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Verify Python path: `python -c "import sys; print(sys.path)"`

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
python app.py --port 5001
```

### Issue: "Milvus collection not found"

**Solution:**
1. Import data first: `python data/import-milvus.py`
2. Verify collection exists:
   ```python
   from pymilvus import Collection, connections
   connections.connect("default", host="localhost", port=19530)
   collections = connections.list_collections()
   print(collections)
   ```

## Security Notes

### Critical: Remove Hardcoded Secrets

✅ **Completed** - All hardcoded secrets have been removed and replaced with environment variables.

### Best Practices

1. **Environment Variables**
   - Use `.env` file (Git-ignored)
   - Never commit secrets to version control
   - Use `.env.example` for documentation

2. **API Key Management**
   - Rotate keys regularly
   - Use different keys for dev/staging/production
   - Implement key expiration
   - Monitor API usage

3. **Network Security**
   - Use HTTPS in production
   - Implement API rate limiting
   - Add authentication/authorization
   - Validate all user inputs

4. **Database Security**
   - Use Milvus authentication (if available)
   - Restrict network access
   - Regular backups
   - Monitor for unauthorized access

5. **Deployment Security**
   - Use secrets manager (Azure Key Vault, AWS Secrets Manager)
   - Container image scanning for vulnerabilities
   - Minimal base images
   - Run as non-root user

### Example Production Setup with Azure Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://<vault-name>.vault.azure.net/", credential=credential)

api_key = client.get_secret("AZURE_OPENAI_API_KEY").value
endpoint = client.get_secret("AZURE_OPENAI_ENDPOINT").value
```

## Next Steps

1. **Customize Models**: Update embedding models based on your use case
2. **Add Authentication**: Implement user authentication for the API
3. **Implement Caching**: Add Redis for response caching
4. **Monitoring**: Add logging and monitoring (e.g., Application Insights)
5. **Testing**: Create comprehensive unit and integration tests
6. **CI/CD**: Set up automated deployment pipeline

## Contributing

To contribute to this project:
1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request with detailed description

## License

See LICENSE file for details.

## Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Review the API documentation above
3. Check logs for error messages
4. Consult Azure OpenAI documentation
5. Check Milvus documentation

## Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Milvus Documentation](https://milvus.io/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [BERT Model Documentation](https://huggingface.co/transformers/model_doc/bert.html)

---

**Last Updated**: May 14, 2026

**Status**: ✅ Production Ready (with proper environment configuration)
