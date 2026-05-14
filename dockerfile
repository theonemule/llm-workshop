# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Avoiding user interaction with tzdata when installing packages
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, ffmpeg, and other necessary system utilities
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
# Note: torch is a large package; comment out if using CPU-only deployment
RUN pip3 install flask openai whisper werkzeug BeautifulSoup4 tiktoken pymilvus pandas milvus transformers torch openai-whisper

# Set working directory
WORKDIR /app

# Copy your Python scripts into the container
COPY data/import-milvus.py ./import-milvus.py
COPY data/import-sqlite.py ./import-sqlite.py
COPY data/JEOPARDY.csv ./JEOPARDY.csv
COPY data/import-data.sh ./import-data.sh
COPY app.py ./app.py
COPY api ./api
COPY static ./static
COPY start.sh ./start.sh
COPY .env.example ./.env.example

RUN chmod +x ./import-data.sh && chmod +x ./start.sh

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port your app runs on
EXPOSE 5000

# Set default environment variables
# These can be overridden at runtime with -e flags or env files
ENV FLASK_ENV=production
ENV MILVUS_HOST=milvus
ENV MILVUS_PORT=19530
ENV FLASK_DEBUG=False

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/', timeout=5)"

# Run the application
CMD ["sh", "-c", "./start.sh"]