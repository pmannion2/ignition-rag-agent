#!/bin/bash
# Script to run a complete demo of the Ignition RAG with Cursor integration

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}           Ignition RAG Cursor Demo              ${NC}"
echo -e "${BLUE}==================================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
  exit 1
fi

# Check if UV is installed
if ! command -v uv &> /dev/null; then
  echo -e "${YELLOW}Warning: UV is not installed. Using regular Python instead.${NC}"
  USE_UV=false
else
  USE_UV=true
fi

# 1. Start the Chroma and API services
echo -e "${YELLOW}Starting Chroma and API services...${NC}"
./run_local.sh &

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10  # Give some time for services to initialize

# 2. Run the sample indexer to populate the database
echo -e "${YELLOW}Populating the database with sample data...${NC}"
if [ "$USE_UV" = true ]; then
  uv run python sample_indexer.py
else
  python sample_indexer.py
fi

# 3. Install the Cursor extension
echo -e "${YELLOW}Installing the Cursor extension...${NC}"
./install_cursor_extension.sh

# 4. Show instructions
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Ignition RAG Cursor Demo is ready!              ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e ""
echo -e "${BLUE}What's running:${NC}"
echo -e "1. Chroma database at ${BLUE}http://localhost:8000${NC}"
echo -e "2. RAG API service at ${BLUE}http://localhost:8001${NC}"
echo -e "3. Cursor extension has been installed to ${BLUE}~/.cursor/extensions/ignition-rag/${NC}"
echo -e ""
echo -e "${BLUE}Try these sample queries in Cursor:${NC}"
echo -e "- How is the Tank1Level tag configured?"
echo -e "- What is the pump status component?"
echo -e "- How do I set up a tank level indicator in Ignition?"
echo -e "- Explain the pump control view"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for user to press Ctrl+C
wait 