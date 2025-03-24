#!/bin/bash
set -e

# Process command line arguments
SKIP_LINT=false
INCLUDE_E2E=false
for arg in "$@"; do
  case $arg in
    --no-lint)
      SKIP_LINT=true
      shift # Remove --no-lint from processing
      ;;
    --include-e2e)
      INCLUDE_E2E=true
      shift # Remove --include-e2e from processing
      ;;
    *)
      # Unknown option
      ;;
  esac
done

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
export MOCK_EMBEDDINGS="true"
export TEST_MODE="true"
export LOG_LEVEL="ERROR"  # Reduce logging noise during tests

# Determine test targets
if [ "$INCLUDE_E2E" = true ]; then
  TEST_TARGETS="tests"
  echo -e "${YELLOW}Running ALL tests including E2E tests - make sure API server is running!${NC}"
else
  TEST_TARGETS="tests/unit tests/integration"
  echo -e "${YELLOW}Running unit and integration tests only. Use --include-e2e to run E2E tests.${NC}"
fi

# Run tests
echo -e "${BLUE}Running tests...${NC}"
if pytest -v --cov=. --cov-report term-missing --cov-config=.coveragerc $TEST_TARGETS; then
  echo -e "${GREEN}All tests passed${NC}"
else
  echo -e "${RED}Tests failed${NC}"
  FAIL=true
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