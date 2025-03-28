# Ignition RAG Agent

A Retrieval-Augmented Generation (RAG) system for Ignition SCADA projects with Cursor IDE integration.

## Overview

This system indexes Ignition project files, creates semantic embeddings, and provides an API to query this information. It includes a Cursor IDE extension that enhances AI capabilities with context from your Ignition projects.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/FastAPI-0.95+-green.svg" alt="FastAPI 0.95+">
  <img src="https://img.shields.io/badge/Cursor-Integration-purple.svg" alt="Cursor Integration">
  <img src="https://img.shields.io/badge/ChromaDB-0.4+-orange.svg" alt="ChromaDB 0.4+">
</p>

## Features

- **Semantic Search**: Find relevant information in your Ignition project files
- **Contextual Responses**: Get AI responses that understand your specific Ignition configuration
- **Cursor Integration**: Use the RAG system directly within Cursor IDE
- **Docker Ready**: Easy deployment with Docker Compose
- **OpenAI API**: Leverages OpenAI's embedding and chat models

## Quick Start

See the [Quick Start Guide](docs/QUICK_START.md) for rapid setup instructions.

### 1. Start the RAG System

```bash
# Clone the repository
git clone https://github.com/yourusername/ignition-rag-agent.git
cd ignition-rag-agent

# Set up environment
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Start services
docker-compose up -d

# Index your Ignition project
docker-compose --profile indexer up
```

### 2. Install Cursor Extension

```bash
# Run the installation script
./install_cursor_extension.sh

# Fix the API URL
echo 'RAG_API_URL=http://localhost:8000' > ~/.cursor/extensions/ignition-rag/.env
```

### 3. Use in Cursor

1. Restart Cursor
2. Use "Ask" (Alt+A) or "Agent" (Alt+Shift+A) 
3. Query about your Ignition project

## Documentation

- [Complete Setup Guide](docs/COMPLETE_SETUP_GUIDE.md) - Detailed installation and configuration
- [API Documentation](docs/API_DOCUMENTATION.md) - API endpoints and usage examples
- [Cursor Extension Guide](docs/CURSOR_README.md) - Cursor IDE integration details
- [Quick Start Guide](docs/QUICK_START.md) - Fast setup instructions

## System Components

- **Indexer**: Processes Ignition JSON files, chunks them, and creates embeddings
- **ChromaDB**: Vector database for storing and searching embeddings
- **FastAPI Service**: API for querying the database 
- **Cursor Extension**: Cursor IDE integration

## Docker Services

| Service | Description | Port |
|---------|-------------|------|
| api | FastAPI service | 8000 |
| chroma | ChromaDB vector database | 8001 |
| indexer | Document indexing (on-demand) | - |
| watcher | File change monitoring (optional) | - |

## API Endpoints

Basic endpoints for using the RAG API:

- `GET /health` - Check system health
- `GET /stats` - Get database statistics
- `POST /query` - Vector search
- `POST /agent/query` - Agent-optimized search
- `POST /agent/chat` - Conversational responses

See the [API Documentation](docs/API_DOCUMENTATION.md) for details.

## Cursor Integration

The Cursor extension allows you to:

1. Enhance AI completions with Ignition-specific context
2. Get agentic responses that understand your Ignition project
3. Use natural language to query your Ignition configuration

See the [Cursor Extension Guide](docs/CURSOR_README.md) for more information.

## Example Usage

Using the API directly:

```bash
# Query the API
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the Tank Level tag configuration",
    "top_k": 3,
    "filter_type": "tag"
  }'
```

Using the Python client:

```python
from cursor_client import get_chat_response

# Get a response
response = get_chat_response(
    query="Explain the Tank Level tag configuration",
    top_k=3,
    filter_type="tag"
)
print(response)
```

## Development

### Running Tests

```bash
# Run unit tests
./run_tests.sh

# Run end-to-end tests with Docker
./run_e2e_tests.sh
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API locally
python -m uvicorn api:app --reload
```

## License

MIT

## Acknowledgements

- [OpenAI](https://openai.com/) for embedding and language models
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Cursor](https://cursor.sh/) for the AI-powered IDE 