# Complete Setup Guide for Ignition RAG Agent with Cursor Integration

This guide provides comprehensive instructions for setting up the Ignition RAG Agent system and integrating it with Cursor IDE for enhanced development experiences.

## Table of Contents

1. [System Overview](#system-overview)
2. [RAG API Setup](#rag-api-setup)
3. [Docker Configuration](#docker-configuration)
4. [Cursor Extension](#cursor-extension)
5. [API Documentation](#api-documentation)
6. [Troubleshooting](#troubleshooting)

## System Overview

The Ignition RAG Agent is a Retrieval-Augmented Generation system for Ignition SCADA projects. It indexes Ignition project files (Perspective views, Tags, etc.), creates semantic embeddings, and provides an API to query this information. The system enhances Cursor IDE's AI capabilities by providing relevant context from your Ignition projects.

**Components:**
- **Indexer**: Processes Ignition project files, chunks them, and generates embeddings
- **ChromaDB**: Vector database for storing and searching embeddings
- **FastAPI Service**: Provides API endpoints for querying the database
- **Cursor Extension**: Integrates with Cursor IDE for enhanced AI capabilities

## RAG API Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenAI API Key
- Ignition project files to index

### Setting Up the Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ignition-rag-agent.git
   cd ignition-rag-agent
   ```

2. **Create a virtual environment (optional for local development):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a .env file with your configuration:**
   ```
   # OpenAI API Key for generating embeddings
   OPENAI_API_KEY=your_openai_key_here
   
   # ChromaDB configuration
   CHROMA_HOST=localhost
   CHROMA_PORT=8001
   
   # Concurrency and rate limits
   OPENAI_TOKENS_PER_MINUTE=50000
   QUERY_CONCURRENCY_LIMIT=10
   INDEX_CONCURRENCY_LIMIT=5
   
   # Timeouts for operations (in seconds)
   DEFAULT_QUERY_TIMEOUT=30
   DEFAULT_INDEX_TIMEOUT=60
   
   # CPU thread pool size for parallel operations
   CPU_THREAD_POOL_SIZE=10
   ```

## Docker Configuration

The recommended way to run the Ignition RAG system is using Docker, which handles all dependencies and services.

### Starting the Services

1. **Start the API and ChromaDB services:**
   ```bash
   docker-compose up -d
   ```
   This starts:
   - ChromaDB on port 8001
   - API service on port 8000

2. **Index your Ignition project files:**
   ```bash
   # Update the IGNITION_PROJECT_PATH in .env file to point to your project
   IGNITION_PROJECT_PATH=/path/to/your/ignition_project
   
   # Run the indexer
   docker-compose --profile indexer up
   ```

3. **Optional: Start the watcher for automatic indexing:**
   ```bash
   docker-compose --profile watcher up -d
   ```

### Docker Container Structure

- **api**: FastAPI service for querying the vector database
- **chroma**: ChromaDB vector database for storing embeddings
- **indexer**: Service for indexing Ignition project files (run on-demand)
- **watcher**: Service for monitoring file changes and triggering re-indexing (optional)

### Ports and Configuration

- **ChromaDB**: Runs on port 8001 (external) / 8000 (internal)
- **API**: Runs on port 8000

## Cursor Extension

The Cursor extension connects the RAG system to Cursor IDE, enhancing AI capabilities with Ignition-specific knowledge.

### Installation

1. **Run the installation script:**
   ```bash
   ./install_cursor_extension.sh
   ```
   This script:
   - Creates the extension directory at `~/.cursor/extensions/ignition-rag/`
   - Sets up a Python virtual environment with dependencies
   - Configures the extension to connect to your API service

2. **Configure the extension:**
   The installation creates a `.env` file at `~/.cursor/extensions/ignition-rag/.env` with:
   ```
   RAG_API_URL=http://localhost:8000
   PYTHON_PATH=/path/to/python
   ```
   Ensure the `RAG_API_URL` points to your API service.

3. **Restart Cursor to enable the extension.**

### Using the Extension

The extension works automatically in the background:

1. Make sure your Docker containers (API and ChromaDB) are running
2. Open Cursor IDE and work on your project
3. Use Cursor's AI features (Alt+A for Ask, Alt+Shift+A for Agent)
4. When you ask questions about Ignition, the extension will automatically enhance the AI with context from your project

Examples of queries that trigger the extension:
- "How do I use the Tank Level tag?"
- "Explain the Perspective view structure"
- "Generate code to display the alarm status"

### Manual Testing

You can test the extension manually:

```bash
~/.cursor/extensions/ignition-rag/run_client.sh "What is the configuration of the Tank Level tag?"
```

This should return relevant information from your indexed Ignition project.

## API Documentation

The Ignition RAG system provides several API endpoints for different use cases.

### Health and Status

- **GET /health**: Check API health and status
  ```bash
  curl http://localhost:8000/health
  ```

- **GET /stats**: Get database statistics
  ```bash
  curl http://localhost:8000/stats
  ```

### Query Endpoints

- **POST /query**: Basic vector search
  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Tank Level configuration", "top_k": 3}'
  ```

- **POST /agent/query**: Enhanced query for agents (used by Cursor)
  ```bash
  curl -X POST http://localhost:8000/agent/query \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Tank Level configuration",
      "top_k": 3,
      "filter_type": "tag",
      "context": {
        "current_file": "path/to/current/file.js"
      }
    }'
  ```

- **POST /agent/chat**: Conversational endpoint with LLM responses
  ```bash
  curl -X POST http://localhost:8000/agent/chat \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Explain the Tank Level tag configuration",
      "top_k": 3,
      "filter_type": "tag",
      "context": {
        "current_file": "path/to/current/file.js"
      }
    }'
  ```

### Response Format

The `/agent/query` endpoint returns:
```json
{
  "context_chunks": [
    {
      "content": "...",
      "source": "tag_file.json",
      "metadata": {
        "filepath": "path/to/tag_file.json",
        "type": "tag"
      },
      "similarity": 0.92
    }
  ],
  "suggested_prompt": "..."
}
```

The `/agent/chat` endpoint returns:
```json
{
  "response": "The Tank Level tag is configured as...",
  "context_chunks": [
    {
      "content": "...",
      "source": "tag_file.json",
      "metadata": {
        "filepath": "path/to/tag_file.json", 
        "type": "tag"
      },
      "similarity": 0.92
    }
  ]
}
```

## Troubleshooting

### Docker Issues

- **Containers won't start:**
  ```bash
  # Check container logs
  docker-compose logs
  
  # Ensure ports are available
  lsof -i :8000
  lsof -i :8001
  ```

- **API can't connect to ChromaDB:**
  Make sure the `CHROMA_HOST` is set to `chroma` in the `.env` file when running in Docker.

### Cursor Extension Issues

- **Extension not loading:**
  1. Check if the extension directory exists:
     ```bash
     ls -la ~/.cursor/extensions/ignition-rag/
     ```
  2. Verify Cursor is allowed to run the scripts:
     ```bash
     chmod +x ~/.cursor/extensions/ignition-rag/run_client.sh
     ```

- **Extension can't connect to API:**
  1. Make sure API is running:
     ```bash
     curl http://localhost:8000/health
     ```
  2. Verify the correct URL in the extension's `.env` file:
     ```bash
     cat ~/.cursor/extensions/ignition-rag/.env
     ```
  3. Update if needed:
     ```bash
     echo 'RAG_API_URL=http://localhost:8000' > ~/.cursor/extensions/ignition-rag/.env
     ```

- **Python errors in extension:**
  1. Check if virtual environment is correctly set up:
     ```bash
     ls -la ~/.cursor/extensions/ignition-rag/venv/bin/
     ```
  2. Reinstall the extension if needed:
     ```bash
     ./install_cursor_extension.sh
     ```

### Common Issues

- **No results returned:** Ensure you've indexed your Ignition project first
- **Slow responses:** Check rate limiting and concurrency settings in the `.env` file
- **"No module found" errors:** Ensure all dependencies are installed 