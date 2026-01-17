#!/bin/bash

# Code Quality Script
# Run formatting and quality checks for the codebase

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Code Quality Checks"
echo "=================================="

# Parse arguments
CHECK_ONLY=false
FORMAT_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=true ;;
        --format) FORMAT_ONLY=true ;;
        -h|--help)
            echo "Usage: ./quality.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check    Check formatting without making changes"
            echo "  --format   Format code (default behavior)"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run black formatter
if [ "$CHECK_ONLY" = true ]; then
    echo -e "\n${YELLOW}Checking code formatting with black...${NC}"
    if uv run black --check backend/ main.py; then
        echo -e "${GREEN}✓ All files are properly formatted${NC}"
    else
        echo -e "${RED}✗ Some files need formatting. Run './quality.sh --format' to fix.${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}Formatting code with black...${NC}"
    uv run black backend/ main.py
    echo -e "${GREEN}✓ Code formatted successfully${NC}"
fi

echo ""
echo "=================================="
echo -e "${GREEN}Quality checks complete!${NC}"
echo "=================================="
