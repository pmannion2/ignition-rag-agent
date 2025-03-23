# Ignition RAG Extension for Cursor

This extension integrates the Ignition RAG (Retrieval-Augmented Generation) system with the Cursor IDE. It enhances Cursor's AI capabilities by providing relevant context from your Ignition project when you ask questions or generate code.

## Features

- Automatically enhances AI prompts with relevant context from your Ignition project
- Works with Perspective views, Tags, and other Ignition project components
- Seamlessly integrates with Cursor's AI features
- Configurable through settings or environment variables

## Installation

### Option 1: Using the installation script

1. Run the installation script:
   ```bash
   ./install_cursor_extension.sh
   ```

2. Restart Cursor to enable the extension

### Option 2: Manual installation

1. Create an extension directory:
   ```bash
   mkdir -p ~/.cursor/extensions/ignition-rag
   ```

2. Copy the extension files:
   ```bash
   cp cursor_extension.js cursor_client.py cursor.config.json ~/.cursor/extensions/ignition-rag/
   ```

3. Create a .env file in the extension directory:
   ```
   RAG_API_URL=http://localhost:8001
   PYTHON_PATH=python3
   ```

4. Restart Cursor to enable the extension

## Usage

The extension works automatically in the background, enhancing AI prompts related to Ignition. Simply:

1. Make sure your Ignition RAG API is running
2. Use Cursor's AI features as normal, but mention Ignition-related terms
3. The extension will automatically fetch relevant context from your Ignition project

Example prompts that will trigger the extension:
- "How do I use this Ignition tag?"
- "Explain the view structure of this perspective component"
- "Write code that interacts with the Ignition tank level"

## Configuration

You can configure the extension by editing the `.env` file in the extension directory:

```
# Ignition RAG Extension Configuration
RAG_API_URL=http://localhost:8001  # URL of your RAG API
PYTHON_PATH=python3                # Path to Python interpreter
```

## Requirements

- Python 3.8+
- Cursor IDE
- Running Ignition RAG API service

## Troubleshooting

If the extension is not working as expected:

1. Make sure the Ignition RAG API is running and accessible
2. Check the Cursor console for error messages (Help > Toggle Developer Tools)
3. Verify that Python is installed and accessible
4. Make sure the extension is correctly installed in `~/.cursor/extensions/ignition-rag/`

## License

MIT 