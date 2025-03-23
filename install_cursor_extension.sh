#!/bin/bash
# Installation script for Ignition RAG Cursor Extension

set -e

# Configuration
EXTENSION_NAME="ignition-rag"
CURSOR_EXTENSIONS_DIR="$HOME/.cursor/extensions"
EXTENSION_DIR="$CURSOR_EXTENSIONS_DIR/$EXTENSION_NAME"
VENV_DIR="$EXTENSION_DIR/venv"

# Check if cursor extension directory exists
if [ ! -d "$CURSOR_EXTENSIONS_DIR" ]; then
  echo "Creating Cursor extensions directory at $CURSOR_EXTENSIONS_DIR"
  mkdir -p "$CURSOR_EXTENSIONS_DIR"
fi

# Remove existing extension if it exists
if [ -d "$EXTENSION_DIR" ]; then
  echo "Removing existing extension..."
  rm -rf "$EXTENSION_DIR"
fi

# Create extension directory
echo "Creating extension directory..."
mkdir -p "$EXTENSION_DIR"

# Copy files
echo "Copying extension files..."
cp cursor_extension.js "$EXTENSION_DIR/"
cp cursor_integration.js "$EXTENSION_DIR/"
cp cursor_connector.js "$EXTENSION_DIR/"
cp cursor_client.py "$EXTENSION_DIR/"
cp cursor.config.json "$EXTENSION_DIR/"

# Copy README if available
if [ -f "CURSOR_README.md" ]; then
  cp CURSOR_README.md "$EXTENSION_DIR/README.md"
else
  cp README.md "$EXTENSION_DIR/" 2>/dev/null || echo "No README found, skipping..."
fi

# Create extension.js that auto-loads on Cursor startup
echo "Creating extension loader..."
cat > "$EXTENSION_DIR/extension.js" << EOF
// Ignition RAG Extension Loader
// This file is automatically loaded by Cursor on startup

const { spawn } = require('child_process');
const path = require('path');

// Update the Python path in the extension
try {
  const fs = require('fs');
  const extensionConfig = require('./cursor_extension');
  
  // Use the shell script wrapper to run the Python client
  extensionConfig.configure({
    clientScript: path.join(__dirname, 'run_client.sh'),
    pythonPath: '' // Not needed as the shell script handles this
  });
  
  // Load the connector
  require('./cursor_connector');
  console.log('Ignition RAG Extension loaded successfully');
} catch (error) {
  console.error('Failed to load Ignition RAG Extension:', error.message);
}
EOF

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo "Warning: python3 not found. You need to install Python 3 for the extension to work."
  exit 1
fi

# Install Python dependencies
echo "Setting up Python dependencies..."

# Create a virtual environment for the extension
echo "Creating a virtual environment..."
python3 -m venv "$VENV_DIR" || {
  echo "Failed to create virtual environment. Python venv module might be missing."
  echo "You may need to install it manually:"
  echo "  On macOS: brew install python-venv"
  echo "  On Ubuntu/Debian: apt-get install python3-venv"
  echo ""
  echo "Alternatively, you can install the required packages manually:"
  echo "  pip3 install --user requests python-dotenv"
  echo ""
  echo "Warning: Dependencies not installed automatically!"
  HAS_VENV=false
}

# If venv was created successfully, install dependencies
if [ -d "$VENV_DIR" ]; then
  echo "Installing dependencies in virtual environment..."
  "$VENV_DIR/bin/pip" install requests python-dotenv || {
    echo "Failed to install dependencies in virtual environment."
    echo "You may need to install them manually:"
    echo "  pip3 install --user requests python-dotenv"
  }
  echo "Dependencies installed successfully in virtual environment."
  HAS_VENV=true
fi

# Create .env file for the extension
if [ ! -f "$EXTENSION_DIR/.env" ]; then
  echo "Creating default .env file..."
  cat > "$EXTENSION_DIR/.env" << EOF
# Ignition RAG Extension Configuration
RAG_API_URL=http://localhost:8001
PYTHON_PATH=$([ "$HAS_VENV" = true ] && echo "$VENV_DIR/bin/python3" || echo "python3")
EOF
fi

# Create a shell script to run the client with the virtual environment
cat > "$EXTENSION_DIR/run_client.sh" << EOF
#!/bin/bash
$([ "$HAS_VENV" = true ] && echo "$VENV_DIR/bin/python3" || echo "python3") "$EXTENSION_DIR/cursor_client.py" "\$@"
EOF
chmod +x "$EXTENSION_DIR/run_client.sh"

echo ""
echo "Ignition RAG extension has been installed to $EXTENSION_DIR"
if [ "$HAS_VENV" = true ]; then
  echo "✅ Virtual environment created and dependencies installed successfully."
else
  echo "⚠️  Virtual environment setup failed. You may need to install dependencies manually:"
  echo "   pip3 install --user requests python-dotenv"
fi
echo "To configure the extension, edit $EXTENSION_DIR/.env"
echo "Restart Cursor to enable the extension" 