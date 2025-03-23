#!/bin/bash
set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ignition RAG Agent Test Suite ===${NC}"

# Create a temp directory for test data
TEST_DIR=$(mktemp -d)
echo -e "${BLUE}Created temporary test directory: ${TEST_DIR}${NC}"

# Clean up on exit
function cleanup {
  echo -e "${BLUE}Cleaning up...${NC}"
  rm -rf "${TEST_DIR}"
}
trap cleanup EXIT

# Prepare sample data
echo -e "${BLUE}Setting up test data...${NC}"
mkdir -p "${TEST_DIR}/ignition_project/views"
mkdir -p "${TEST_DIR}/ignition_project/tags"
cp tests/data/sample_view.json "${TEST_DIR}/ignition_project/views/tank_view.json"
cp tests/data/sample_tags.json "${TEST_DIR}/ignition_project/tags/tank_tags.json"

# Set up environment for testing
echo -e "${BLUE}Setting up test environment...${NC}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OPENAI_API_KEY=${OPENAI_API_KEY}
export MOCK_EMBEDDINGS="true"
export TEST_MODE="true"
export LOG_LEVEL="ERROR"  # Reduce logging noise during tests

# Run code quality checks
echo -e "${YELLOW}Running code quality checks...${NC}"

echo -e "Running black..."
if command -v black &> /dev/null; then
  if black --check . ; then
    echo -e "${GREEN}Black check passed${NC}"
  else
    echo -e "${RED}Black check failed${NC}"
    exit 1
  fi
else
  echo -e "${YELLOW}Black not installed, skipping${NC}"
fi

echo -e "Running flake8..."
if command -v flake8 &> /dev/null; then
  if flake8 api.py indexer.py cursor_agent.py watcher.py logger.py example_client.py main.py --count --select=E9,F63,F7,F82 --show-source --statistics; then
    echo -e "${GREEN}Flake8 check passed${NC}"
  else
    echo -e "${RED}Flake8 check failed${NC}"
    exit 1
  fi
else
  echo -e "${YELLOW}Flake8 not installed, skipping${NC}"
fi

# Run unit tests
echo -e "${YELLOW}Running unit tests...${NC}"
python -m pytest tests/unit -v

# Run simplified integration test
echo -e "${YELLOW}Running integration test...${NC}"

# Run indexer on test data
echo -e "Indexing test data..."
python indexer.py --path "${TEST_DIR}/ignition_project" --rebuild

# Start API server in background
echo -e "Starting API server in background..."
API_PORT=8765
python -c "from multiprocessing import Process; from api import app; import uvicorn; Process(target=uvicorn.run, args=(app,), kwargs={'host': '127.0.0.1', 'port': ${API_PORT}, 'log_level': 'error'}).start()" &
API_PID=$!

# Wait for API to start
echo -e "Waiting for API to start..."
sleep 3

# Test the API
echo -e "Testing API endpoints..."
QUERY_RESPONSE=$(curl -s -X POST http://localhost:${API_PORT}/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Tank Level", "top_k": 2}')

HEALTH_RESPONSE=$(curl -s http://localhost:${API_PORT}/health)

# Clean up API server
echo -e "Stopping API server..."
kill $API_PID

# Verify results
if [[ $QUERY_RESPONSE == *"results"* ]]; then
  echo -e "${GREEN}API query test passed${NC}"
else
  echo -e "${RED}API query test failed${NC}"
  echo $QUERY_RESPONSE
  exit 1
fi

if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
  echo -e "${GREEN}API health check passed${NC}"
else
  echo -e "${RED}API health check failed${NC}"
  echo $HEALTH_RESPONSE
  exit 1
fi

# Run Docker tests if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
  echo -e "${YELLOW}Running Docker build test...${NC}"
  docker build -t ignition-rag-test .
  echo -e "${GREEN}Docker build successful${NC}"
else
  echo -e "${YELLOW}Docker not available, skipping Docker tests${NC}"
fi

echo -e "${GREEN}All tests passed successfully!${NC}" 