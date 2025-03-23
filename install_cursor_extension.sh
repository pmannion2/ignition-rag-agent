#!/bin/bash
# Installation script for Ignition RAG Cursor Extension

set -e

# Configuration
EXTENSION_NAME="ignition-rag"
CURSOR_EXTENSIONS_DIR="$HOME/.cursor/extensions"
EXTENSION_DIR="$CURSOR_EXTENSIONS_DIR/$EXTENSION_NAME"

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

try {
  // Load the connector
  require('./cursor_connector');
  console.log('Ignition RAG Extension loaded successfully');
} catch (error) {
  console.error('Failed to load Ignition RAG Extension:', error.message);
}
EOF

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo "Warning: python3 not found. You may need to install Python 3 for the extension to work."
fi

# Check if pip is installed
if command -v pip3 &> /dev/null; then
  echo "Installing Python dependencies..."
  pip3 install requests python-dotenv
else
  echo "Warning: pip3 not found. You may need to install Python dependencies manually."
  echo "Required packages: requests, python-dotenv"
fi

# Create .env file for the extension
if [ ! -f "$EXTENSION_DIR/.env" ]; then
  echo "Creating default .env file..."
  cat > "$EXTENSION_DIR/.env" << EOF
# Ignition RAG Extension Configuration
RAG_API_URL=http://localhost:8001
PYTHON_PATH=python3
EOF
fi

echo "Ignition RAG extension has been installed to $EXTENSION_DIR"
echo "To configure the extension, edit $EXTENSION_DIR/.env"
echo "Restart Cursor to enable the extension" 