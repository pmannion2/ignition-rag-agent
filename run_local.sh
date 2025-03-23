#!/bin/bash
# Script to run the Ignition RAG services locally

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set default variables
USE_MOCK=true
API_PORT=8001
DOCKER_COMPOSE_FILE="docker-compose.local.yml"

# Print header
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}      Ignition RAG Local Development Setup        ${NC}"
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

# Start Chroma DB with Docker
echo -e "${YELLOW}Starting Chroma database...${NC}"
docker-compose -f $DOCKER_COMPOSE_FILE up -d chroma

# Wait for Chroma to be healthy
echo -e "${YELLOW}Waiting for Chroma to be ready...${NC}"
attempt=0
max_attempts=30
until docker-compose -f $DOCKER_COMPOSE_FILE exec chroma curl -s http://localhost:8000/api/v1/heartbeat > /dev/null; do
  attempt=$((attempt+1))
  if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}Error: Chroma failed to start properly.${NC}"
    docker-compose -f $DOCKER_COMPOSE_FILE logs chroma
    docker-compose -f $DOCKER_COMPOSE_FILE down
    exit 1
  fi
  echo -n "."
  sleep 1
done
echo -e "\n${GREEN}Chroma is ready!${NC}"

# Set environment variables for the API
export MOCK_EMBEDDINGS=$USE_MOCK
export CHROMA_HOST=localhost
export CHROMA_PORT=8000

# Start the API service
echo -e "${YELLOW}Starting API service on port $API_PORT...${NC}"
echo -e "${YELLOW}Using mock embeddings: $USE_MOCK${NC}"

# Trap to catch Ctrl+C and shut down services
trap 'echo -e "${YELLOW}Shutting down services...${NC}"; \
      kill $API_PID 2>/dev/null; \
      docker-compose -f $DOCKER_COMPOSE_FILE down; \
      echo -e "${GREEN}All services stopped.${NC}"; \
      exit 0' INT

# Run API with UV or regular Python
if [ "$USE_UV" = true ]; then
  echo -e "${GREEN}Running with UV${NC}"
  uv run uvicorn api:app --reload --port $API_PORT &
  API_PID=$!
else
  echo -e "${YELLOW}Running with regular Python${NC}"
  python -m uvicorn api:app --reload --port $API_PORT &
  API_PID=$!
fi

# Print success message
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Ignition RAG services are running                ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "API is available at: ${BLUE}http://localhost:$API_PORT${NC}"
echo -e "API Docs available at: ${BLUE}http://localhost:$API_PORT/docs${NC}"
echo -e "Chroma UI is available at: ${BLUE}http://localhost:8000/ui${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all services"

# Wait for API process to complete (or be terminated)
wait $API_PID 