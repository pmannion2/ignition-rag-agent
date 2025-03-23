#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Ruff linter on Python files...${NC}"

# Check if inside virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    # Check if .venv exists and activate it
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source .venv/bin/activate
    else
        echo -e "${RED}No virtual environment found. Please create and activate one first.${NC}"
        echo -e "${YELLOW}You can create one with: python -m venv .venv${NC}"
        exit 1
    fi
fi

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Ruff not found, installing...${NC}"
    pip install ruff
fi

# Run Ruff for linting
echo -e "${GREEN}Running Ruff linter...${NC}"
ruff check .

# Run Ruff for formatting check (but don't modify files)
echo -e "${GREEN}Checking formatting with Ruff...${NC}"
ruff format --check .

# If you want to automatically fix issues, uncomment the following lines:
# echo -e "${GREEN}Fixing auto-fixable issues...${NC}"
# ruff check --fix .
# ruff format .

echo -e "${GREEN}Linting completed!${NC}" 