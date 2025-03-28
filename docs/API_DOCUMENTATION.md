# Ignition RAG Agent API Documentation

This document provides detailed specifications for the Ignition RAG API endpoints, request/response formats, and usage examples.

## Base URL

When running locally with Docker Compose, the API is available at:
```
http://localhost:8000
```

## Authentication

Currently, the API does not implement authentication. For production deployments, consider adding an authentication mechanism.

## API Endpoints

### Health and Status

#### GET /health

Check API health and dependencies status.

**Response:**
```json
{
  "status": "ok",
  "chroma_status": "connected",
  "openai_status": "available",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

#### GET /stats

Get statistics about the indexed data.

**Response:**
```json
{
  "total_documents": 1250,
  "total_chunks": 4532,
  "document_types": {
    "perspective": 723,
    "tag": 527
  },
  "embedding_dimensions": 1536,
  "index_last_updated": "2023-03-22T16:45:30Z"
}
```

### Basic Search

#### POST /query

Perform basic vector search against the database.

**Request:**
```json
{
  "query": "Tank Level configuration",
  "top_k": 3,
  "filter": {
    "type": "tag"
  },
  "include_metadata": true
}
```

**Parameters:**
- `query` (string, required): The natural language query to search for
- `top_k` (integer, optional): Number of results to return (default: 5)
- `filter` (object, optional): Filters to apply to the search
- `include_metadata` (boolean, optional): Whether to include metadata in results (default: true)

**Response:**
```json
{
  "results": [
    {
      "content": "...",
      "source": "tag_file.json",
      "metadata": {
        "filepath": "path/to/tag_file.json",
        "type": "tag",
        "modified_date": "2023-03-21T14:30:00Z"
      },
      "similarity": 0.92
    },
    {
      "content": "...",
      "source": "another_file.json",
      "metadata": {
        "filepath": "path/to/another_file.json",
        "type": "tag",
        "modified_date": "2023-03-20T10:15:00Z"
      },
      "similarity": 0.87
    }
  ]
}
```

### Agent API

These endpoints are optimized for integration with AI agents like the Cursor extension.

#### POST /agent/query

Get context and suggested prompt for an agent query.

**Request:**
```json
{
  "query": "Tank Level configuration",
  "top_k": 3,
  "filter_type": "tag",
  "context": {
    "current_file": "path/to/current/file.js",
    "selected_text": "tank.level"
  }
}
```

**Parameters:**
- `query` (string, required): The natural language query to search for
- `top_k` (integer, optional): Number of results to return (default: 5)
- `filter_type` (string, optional): Filter by document type (perspective or tag)
- `context` (object, optional): Contextual information about the current state
  - `current_file` (string, optional): Path to the current file being edited
  - `selected_text` (string, optional): Text currently selected in the editor

**Response:**
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
  "suggested_prompt": "Based on the Ignition project files, the Tank Level tag is configured as follows:\n\n[context details]"
}
```

#### POST /agent/chat

Get a conversational LLM response with RAG context.

**Request:**
```json
{
  "query": "Explain the Tank Level tag configuration",
  "top_k": 3,
  "filter_type": "tag",
  "context": {
    "current_file": "path/to/current/file.js"
  },
  "conversation_history": [
    {"role": "user", "content": "Tell me about Ignition tags"},
    {"role": "assistant", "content": "Ignition tags are..."}
  ]
}
```

**Parameters:**
- `query` (string, required): The natural language query to search for
- `top_k` (integer, optional): Number of results to return (default: 5)
- `filter_type` (string, optional): Filter by document type (perspective or tag)
- `context` (object, optional): Contextual information about the current state
- `conversation_history` (array, optional): Previous messages in the conversation

**Response:**
```json
{
  "response": "The Tank Level tag is configured as a float-type tag with engineering units set to 'meters'. It has an alarm configured for high level at 3.5 meters and a critical high alarm at 4.0 meters. The tag is connected to the OPC-UA server with address 'ns=1;s=Device1.TankLevel'.",
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

### Indexing API

These endpoints manage the document indexing process.

#### POST /index

Trigger indexing of specified documents.

**Request:**
```json
{
  "path": "/path/to/ignition_project",
  "rebuild": false,
  "changed_only": true,
  "specific_files": ["file1.json", "file2.json"]
}
```

**Parameters:**
- `path` (string, required): Path to the Ignition project
- `rebuild` (boolean, optional): Whether to rebuild the entire index (default: false)
- `changed_only` (boolean, optional): Only index files that have changed (default: false)
- `specific_files` (array, optional): List of specific files to index

**Response:**
```json
{
  "status": "success",
  "indexed_files": 2,
  "skipped_files": 0,
  "elapsed_time_seconds": 3.45
}
```

#### GET /index/status

Get the current status of the indexing process.

**Response:**
```json
{
  "status": "idle",
  "last_indexed_time": "2023-03-22T14:30:00Z",
  "total_documents": 1250,
  "completed_percentage": 100
}
```

## Error Handling

All API endpoints use standard HTTP status codes:

- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found
- 500: Internal Server Error

Error responses follow this format:

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "The parameter 'query' is required",
    "details": {}
  }
}
```

## Rate Limiting

The API implements rate limiting based on concurrent operations:

- Query operations: Limited by `QUERY_CONCURRENCY_LIMIT` (default: 10)
- Index operations: Limited by `INDEX_CONCURRENCY_LIMIT` (default: 5)
- OpenAI token usage: Limited by `OPENAI_TOKENS_PER_MINUTE` (default: 150,000)

When rate limits are reached, requests will be queued rather than rejected.

## Example Usage

### Curl Examples

**Basic Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tank Level configuration", "top_k": 3}'
```

**Agent Query:**
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

**Chat Response:**
```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the Tank Level tag configuration",
    "top_k": 3,
    "filter_type": "tag"
  }'
```

### Python Examples

**Using the requests library:**
```python
import requests

# Basic query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Tank Level configuration",
        "top_k": 3
    }
)
print(response.json())

# Agent chat
response = requests.post(
    "http://localhost:8000/agent/chat",
    json={
        "query": "Explain the Tank Level tag configuration",
        "top_k": 3,
        "filter_type": "tag"
    }
)
print(response.json()["response"])
```

**Using the built-in client:**
```python
from cursor_client import get_rag_context, get_chat_response

# Get context
context = get_rag_context(
    query="Tank Level configuration", 
    top_k=3, 
    filter_type="tag"
)
print(context)

# Get chat response
response = get_chat_response(
    query="Explain the Tank Level tag configuration", 
    top_k=3, 
    filter_type="tag"
)
print(response)
``` 