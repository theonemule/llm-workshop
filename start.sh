#!/bin/bash

# Exit on error
set -e

echo "================================"
echo "LLM Workshop Application Startup"
echo "================================"

# Wait for Milvus to be ready
if [ -n "$MILVUS_HOST" ]; then
    echo "Waiting for Milvus at $MILVUS_HOST:$MILVUS_PORT..."
    for i in {1..30}; do
        if python3 -c "from pymilvus import connections; connections.connect('default', host='$MILVUS_HOST', port=$MILVUS_PORT); print('Connected')" 2>/dev/null; then
            echo "✓ Milvus is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "✗ Timeout waiting for Milvus"
            exit 1
        fi
        echo "  Attempt $i/30 - Waiting..."
        sleep 2
    done
fi

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Verify required environment variables
echo "Verifying environment configuration..."
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "⚠ WARNING: AZURE_OPENAI_API_KEY not set"
fi
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "⚠ WARNING: AZURE_OPENAI_ENDPOINT not set"
fi

echo "Starting Flask application..."
echo "  Flask Environment: ${FLASK_ENV:-development}"
echo "  Flask Debug: ${FLASK_DEBUG:-False}"
echo "  Milvus Host: ${MILVUS_HOST:-localhost}"
echo "  Milvus Port: ${MILVUS_PORT:-19530}"

# Start the application
python3 app.py

