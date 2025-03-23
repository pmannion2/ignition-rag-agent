# Ignition RAG Agent

A Retrieval-Augmented Generation (RAG) pipeline for indexing and querying Ignition project files.

[![CI/CD](https://github.com/yourusername/ignition-rag-agent/actions/workflows/main.yml/badge.svg)](https://github.com/yourusername/ignition-rag-agent/actions/workflows/main.yml)

## Overview

This system indexes Ignition Perspective views and Tag configurations (stored as JSON files) for semantic search. It chunks these files, generates embeddings using OpenAI's embedding model, and stores them in a Chroma vector database. A FastAPI service allows querying this database to retrieve relevant context.

## 🚀 Setup

### Local Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the indexer to build the vector database:
   ```bash
   python indexer.py --path /path/to/ignition_project
   ```

4. Start the FastAPI service:
   ```bash
   uvicorn api:app --reload
   ```

### Docker Setup

We provide Docker configurations to easily deploy the entire system:

1. Make sure you have Docker and Docker Compose installed.

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   IGNITION_PROJECT_PATH=/path/to/your/ignition_project
   ```

3. Start the API service with Chroma:
   ```bash
   docker-compose up -d
   ```

4. Run the indexer to populate the database:
   ```bash
   docker-compose --profile indexer up
   ```

5. (Optional) Start the watcher for automatic re-indexing:
   ```bash
   docker-compose --profile watcher up -d
   ```

## 📝 Usage

### Indexing Commands

- Full rebuild: `python indexer.py --path /path/to/ignition_project --rebuild`
- Index changed files: `python indexer.py --path /path/to/ignition_project --changed-only`
- Index specific file: `python indexer.py --path /path/to/ignition_project --file specific_file.json`

### API Endpoints

#### Standard API
- `GET /health`: Check API health and dependencies status
- `GET /stats`: Get statistics about the indexed data
- `POST /query`: Query the vector database
  ```json
  {
    "query": "How is the Tank Level tag configured?",
    "top_k": 5
  }
  ```

#### Agent API for Cursor Integration
- `POST /agent/query`: Agent-optimized query endpoint
  ```json
  {
    "query": "How is the Tank Level tag configured?",
    "top_k": 5,
    "filter_type": "tag",  
    "context": {
      "current_file": "path/to/current/file.js"
    }
  }
  ```

### Running with Different Environments

You can configure different environments using environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHROMA_HOST`, `CHROMA_PORT`: For connecting to a remote Chroma server
- `LOG_LEVEL`: Set logging level (INFO, DEBUG, etc.)
- `DEBUG`: Enable debug mode ("true" or "false")
- `API_PORT`: Change the API port (default: 8000)

## 🔌 Cursor Integration

This project includes comprehensive integration with Cursor IDE to enhance code completion and AI responses with context from your Ignition project.

### Cursor Extension

We provide a full Cursor extension that seamlessly integrates with the IDE:

1. Install the extension:
   ```bash
   ./install_cursor_extension.sh
   ```

2. Restart Cursor to enable the extension

3. Use Cursor's AI features with Ignition-related prompts to automatically retrieve context

The extension includes:
- Automatic enhancement of Ignition-related prompts with relevant context
- Integration with Cursor's agent system
- Configuration options via `.env` file

See the [Cursor Extension README](CURSOR_README.md) for more details.

### Python Integration

Use the `cursor_agent.py` module in your Python Cursor workflows:

```python
from cursor_agent import get_cursor_context

# Get context for a query
context = get_cursor_context(
    "How is the Tank Level tag configured?",
    cursor_context={"current_file": "path/to/file.js"}
)

# Use in your Cursor agent interactions
```

### JavaScript Integration

For JavaScript-based Cursor Agent integration, use `cursor_integration.js`:

```javascript
const { enhanceAgentCommand } = require('./cursor_integration');

// Register as a command enhancer
cursor.registerCommandEnhancer(async (command, context) => {
  return await enhanceAgentCommand(command, {
    currentFile: context.currentFile,
    language: context.language,
  });
});
```

### Testing the Integration

To quickly test the integration without restarting Cursor:

```bash
# Make test script executable
chmod +x test_integration.js

# Run the test script
./test_integration.js
```

## 🧪 Testing

### Running Unit Tests

Run the included unit and integration tests:
```bash
./run_tests.sh
```

### End-to-End Testing with Docker

The project includes end-to-end tests that validate the full application stack using Docker:

1. Ensure Docker and docker-compose are installed on your system.

2. Run the E2E test suite:
   ```bash
   ./run_e2e_tests.sh
   ```

This will:
- Build all necessary Docker containers
- Start the application stack with test data
- Run E2E tests against the live services
- Clean up containers when done

For CI/CD environments, add the following to your workflow:
```yaml
- name: Run E2E Tests
  run: ./run_e2e_tests.sh
```

## 📊 Performance Monitoring

The API includes performance monitoring with detailed logging and timing information. Logs are stored in the `logs` directory with rotation:

- `logs/app.log`: General application logs
- `logs/error.log`: Error-only logs

## 🏗️ CI/CD Pipeline

We use GitHub Actions for continuous integration and deployment. The pipeline:

1. Runs unit and integration tests
2. Checks code quality with linters
3. Builds and pushes Docker images (for main branch)

## 🛠️ Project Structure

- `indexer.py`: Handles parsing, chunking, and indexing of Ignition JSON files
- `api.py`: FastAPI service for querying the vector database
- `logger.py`: Centralized logging configuration
- `watcher.py`: Optional file watcher for automatic re-indexing
- `cursor_agent.py`: Python module for Cursor integration
- `cursor_integration.js`: JavaScript integration for Cursor
- `cursor_extension.js`: Cursor extension implementation
- `cursor_client.py`: Python client for the Cursor extension
- `cursor_connector.js`: Connector for Cursor's agent system
- `install_cursor_extension.sh`: Installation script for the Cursor extension
- `chroma_index/`: Directory where the vector database is stored
- `docker-compose.yml`: Docker Compose configuration
- `Dockerfile`: Docker configuration
- `tests/`: Test suite

## 🔒 Security

- All external dependencies are verified with proper error handling
- API endpoints include rate limiting and input validation
- Authentication can be added via environment variables

## 📄 License

MIT License - See LICENSE file for details.

## Testing

### Mock Embeddings

The system supports a mock embeddings mode for testing without requiring an OpenAI API key. This is useful for:

- Running tests in CI/CD pipelines
- Local development without API credentials
- Reducing costs during testing

To enable mock embeddings:

```bash
# Set environment variable
export MOCK_EMBEDDINGS=true

# Run tests
python -m pytest
```

In mock embedding mode, instead of making API calls to OpenAI, a fixed embedding vector is returned, allowing all functionality to be tested without actual embedding generation.

The mock implementation is available in both the `indexer.py` and `api.py` modules:

```python
def mock_embedding(text):
    """Mock embedding function that returns a fixed vector."""
    return [0.1] * 1536  # Same dimensionality as OpenAI's text-embedding-ada-002
```

When running the test suite with `run_tests.sh`, mock embeddings are enabled by default.

## 🧪 Development

### Code Linting with Ruff

This project uses [Ruff](https://github.com/astral-sh/ruff) for fast Python linting and formatting.

1. Run the linter:
   ```bash
   ./run_lint.sh
   ```
   
   Or directly with Ruff:
   ```bash
   # Check code for issues
   ruff check .
   
   # Auto-fix issues where possible
   ruff check --fix .
   
   # Check formatting
   ruff format --check .
   
   # Format code
   ruff format .
   ```

2. Pre-commit Hook
   A pre-commit hook is installed to automatically check your code when committing. You can bypass it with `git commit --no-verify` if needed. 