# Ignition RAG + Cursor: Quick Start Guide

This guide provides the fastest path to getting the Ignition RAG system running and connected to Cursor.

## 1. Start the RAG System

Make sure you have Docker and Docker Compose installed.

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/ignition-rag-agent.git
cd ignition-rag-agent

# Set up your OpenAI API key in .env
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "IGNITION_PROJECT_PATH=/path/to/your/ignition_project" >> .env

# Start the core services
docker-compose up -d

# Index your Ignition project
docker-compose --profile indexer up
```

## 2. Install Cursor Extension

```bash
# Run the installation script
./install_cursor_extension.sh

# Fix the API URL (make sure it points to port 8000)
echo 'RAG_API_URL=http://localhost:8000' > ~/.cursor/extensions/ignition-rag/.env
echo 'PYTHON_PATH=/Users/$(whoami)/.cursor/extensions/ignition-rag/venv/bin/python3' >> ~/.cursor/extensions/ignition-rag/.env

# Ensure the script is executable
chmod +x ~/.cursor/extensions/ignition-rag/run_client.sh
```

## 3. Use in Cursor

1. Restart Cursor
2. Use "Ask" (Alt+A) or "Agent" (Alt+Shift+A) 
3. Query something about your Ignition project:
   - "How is the Tank Level tag configured?"
   - "Explain the Perspective view structure"
   - "Help me understand the alarm configuration"

## Verify It's Working

```bash
# Test the client directly
~/.cursor/extensions/ignition-rag/run_client.sh "What is Ignition SCADA?"
```

## Common Issues

- **No results?** Make sure you've indexed your project
- **Connection errors?** Ensure Docker containers are running (`docker ps`)
- **Wrong port?** Make sure the RAG_API_URL points to port 8000

For detailed setup and troubleshooting, see [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) 