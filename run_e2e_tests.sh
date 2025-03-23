#!/bin/bash
set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ignition RAG Agent E2E Test Suite ===${NC}"

# Check if Docker is available
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker and docker-compose are required for E2E tests${NC}"
    exit 1
fi

# Clean up function
function cleanup {
    echo -e "${BLUE}Cleaning up containers...${NC}"
    docker-compose -f docker-compose.e2e.yml down -v
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Build and start containers
echo -e "${BLUE}Building Docker containers for E2E tests...${NC}"
docker-compose -f docker-compose.e2e.yml build

# First start the chroma and API services
echo -e "${BLUE}Starting Chroma and API services...${NC}"
docker-compose -f docker-compose.e2e.yml up -d chroma api

# Wait a moment for the service to initialize
echo -e "${BLUE}Waiting for services to initialize...${NC}"
sleep 5

# Run indexer
echo -e "${BLUE}Running indexer...${NC}"
docker-compose -f docker-compose.e2e.yml up --exit-code-from indexer indexer

# Run the E2E tests
echo -e "${BLUE}Running E2E tests...${NC}"
docker-compose -f docker-compose.e2e.yml up --exit-code-from e2e e2e

# Store the exit code
EXIT_CODE=$?

if [ "$EXIT_CODE" == "0" ]; then
    echo -e "${GREEN}E2E tests passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}E2E tests failed with exit code $EXIT_CODE${NC}"
    exit 1
fi 