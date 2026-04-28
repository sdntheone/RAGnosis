#!/bin/bash

echo "🚀 Creating RAGnosis project structure inside current folder..."

# App structure
mkdir -p app/api
mkdir -p app/core
mkdir -p app/ingestion
mkdir -p app/retrieval
mkdir -p app/llm
mkdir -p app/services
mkdir -p app/utils

# Data folders
mkdir -p data/raw
mkdir -p data/processed

# Vector DB
mkdir -p vector_db/faiss_index

# Other folders
mkdir -p notebooks
mkdir -p tests
mkdir -p scripts

# Create __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/ingestion/__init__.py
touch app/retrieval/__init__.py
touch app/llm/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py

# Create main files
touch app/api/routes.py
touch app/core/config.py
touch app/core/constants.py

touch app/ingestion/loader.py
touch app/ingestion/cleaner.py
touch app/ingestion/chunker.py
touch app/ingestion/embedder.py
touch app/ingestion/pipeline.py

touch app/retrieval/vector_store.py
touch app/retrieval/retriever.py

touch app/llm/prompt_template.py
touch app/llm/generator.py

touch app/services/rag_pipeline.py

touch app/utils/logger.py
touch app/utils/helpers.py

# Scripts
touch scripts/ingest.py
touch scripts/rebuild_index.py

# Root files (only if not exist)
touch main.py
touch requirements.txt
touch Dockerfile
touch docker-compose.yml
touch README.md
touch .env

echo "✅ RAGnosis structure created successfully!"