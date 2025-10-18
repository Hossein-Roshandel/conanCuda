#!/bin/bash
# Static analysis script for C++ and CUDA code
# Runs clang-tidy on all source files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Static Analysis with clang-tidy"
echo "=========================================="
echo ""

# Check if compile_commands.json exists
if [ ! -f "build/compile_commands.json" ]; then
    echo -e "${YELLOW}Warning: compile_commands.json not found${NC}"
    echo "Generating with CMake..."
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build
fi

# Find all C++ and CUDA source files
echo "Finding source files..."
FILES=$(find . -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.h" \) \
    ! -path "*/build/*" ! -path "*/.git/*" ! -path "*/docs/*")

if [ -z "$FILES" ]; then
    echo -e "${RED}No source files found${NC}"
    exit 1
fi

echo "Files to analyze:"
echo "$FILES" | sed 's/^/  /'
echo ""

# Run clang-tidy
echo "Running clang-tidy..."
echo ""

FAILED=0
for file in $FILES; do
    echo "Analyzing: $file"
    if clang-tidy -p build "$file" --quiet; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file${NC}"
        FAILED=1
    fi
    echo ""
done

# Summary
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Static analysis passed${NC}"
else
    echo -e "${RED}✗ Static analysis found issues${NC}"
fi
echo "=========================================="

exit $FAILED
