# Ignition RAG Extension for Cursor

This extension integrates the Ignition RAG (Retrieval-Augmented Generation) system with the Cursor IDE. It enhances Cursor's AI capabilities by providing relevant context from your Ignition project when you ask questions or generate code.

## Features

- Automatically enhances AI prompts with relevant context from your Ignition project
- Works with Perspective views, Tags, and other Ignition project components
- Seamlessly integrates with Cursor's AI features
- Configurable through settings or environment variables

## Installation

### Option 1: Using the installation script (Recommended)

1. Run the installation script:
   ```bash
   ./install_cursor_extension.sh
   ```

   The script will:
   - Create a virtual environment for Python dependencies
   - Install required packages in the isolated environment
   - Configure the extension to use the virtual environment

2. Restart Cursor to enable the extension

### Option 2: Manual installation

1. Create an extension directory:
   ```bash
   mkdir -p ~/.cursor/extensions/ignition-rag
   ```

2. Create a virtual environment for Python dependencies:
   ```bash
   python3 -m venv ~/.cursor/extensions/ignition-rag/venv
   ```

3. Install required Python packages:
   ```bash
   ~/.cursor/extensions/ignition-rag/venv/bin/pip install requests python-dotenv
   ```

4. Copy the extension files:
   ```bash
   cp cursor_extension.js cursor_client.py cursor.config.json cursor_integration.js cursor_connector.js ~/.cursor/extensions/ignition-rag/
   ```

5. Create a shell script wrapper for the Python client:
   ```bash
   echo '#!/bin/bash
   ~/.cursor/extensions/ignition-rag/venv/bin/python3 ~/.cursor/extensions/ignition-rag/cursor_client.py "$@"' > ~/.cursor/extensions/ignition-rag/run_client.sh
   chmod +x ~/.cursor/extensions/ignition-rag/run_client.sh
   ```

6. Create a .env file in the extension directory:
   ```
   RAG_API_URL=http://localhost:8001
   PYTHON_PATH=~/.cursor/extensions/ignition-rag/venv/bin/python3
   ```

7. Restart Cursor to enable the extension

## Troubleshooting Installation

### External Environment Errors

If you encounter an error about "externally-managed-environment" during installation, this is because your system Python is protected from modifications. The installation script will handle this by creating a virtual environment, but if you encounter issues:

1. Check that the Python venv module is installed:
   ```bash
   # On macOS with Homebrew
   brew install python-venv
   
   # On Ubuntu/Debian
   sudo apt-get install python3-venv
   ```

2. If you still have issues, manually install the dependencies:
   ```bash
   pip3 install --user requests python-dotenv
   ```

3. Edit the `.env` file to point to your system Python:
   ```
   PYTHON_PATH=python3
   ```

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
PYTHON_PATH=/path/to/python        # Path to Python interpreter (usually automatic)
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
5. Verify the virtual environment was created successfully at `~/.cursor/extensions/ignition-rag/venv/`

## License

MIT 