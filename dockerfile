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
RUN pip3 install flask openai whisper werkzeug BeautifulSoup4 tiktoken pymilvus pandas milvus transformers tensor openai-whisper


# Copy your Python scripts into the container
COPY data/import-milvus.py import-milvus.py
COPY data/import-sqlite.py import-sqlite.py
COPY app.py /app.py

RUN milvus-server & && python3 ./data/import-milvus.py && python3 ./data/import-sqlite.py

# Expose the port your app runs on
EXPOSE 5000


# This CMD instruction might need to be replaced or supplemented by a more complex startup script that can handle starting the server in the background and then executing your Python scripts.
CMD ["sh", "-c", "milvus-server & && python3 app.py"]