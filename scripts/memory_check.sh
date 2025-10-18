#!/bin/bash
# Memory check script for CUDA applications
# Runs both Valgrind (for CPU code) and cuda-memcheck (for GPU code)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXECUTABLE="${1:-./build/build/Release/vectoradd}"
VALGRIND_OPTS="--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose"
CUDA_MEMCHECK_OPTS="--leak-check full --tool memcheck"

echo "=========================================="
echo "Memory Analysis Tools for CUDA"
echo "=========================================="
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable not found: $EXECUTABLE${NC}"
    echo "Please build the project first with: make build"
    exit 1
fi

echo "Target executable: $EXECUTABLE"
echo ""

# Function to run Valgrind
run_valgrind() {
    echo "=========================================="
    echo "Running Valgrind (CPU Memory Check)"
    echo "=========================================="

    if ! command -v valgrind &> /dev/null; then
        echo -e "${YELLOW}Warning: Valgrind not found. Skipping CPU memory check.${NC}"
        echo "Install with: apt-get install valgrind"
        return 1
    fi

    echo "Command: valgrind $VALGRIND_OPTS $EXECUTABLE"
    echo ""

    if valgrind $VALGRIND_OPTS $EXECUTABLE 2>&1 | tee valgrind_report.txt; then
        echo ""
        echo -e "${GREEN}✓ Valgrind completed successfully${NC}"
        echo "Full report saved to: valgrind_report.txt"
    else
        echo ""
        echo -e "${RED}✗ Valgrind detected issues${NC}"
        return 1
    fi
}

# Function to run CUDA memcheck
run_cuda_memcheck() {
    echo ""
    echo "=========================================="
    echo "Running cuda-memcheck (GPU Memory Check)"
    echo "=========================================="

    if ! command -v cuda-memcheck &> /dev/null; then
        echo -e "${YELLOW}Warning: cuda-memcheck not found. Skipping GPU memory check.${NC}"
        echo "cuda-memcheck is part of CUDA toolkit"
        return 1
    fi

    echo "Command: cuda-memcheck $CUDA_MEMCHECK_OPTS $EXECUTABLE"
    echo ""

    if cuda-memcheck $CUDA_MEMCHECK_OPTS $EXECUTABLE 2>&1 | tee cuda_memcheck_report.txt; then
        echo ""
        echo -e "${GREEN}✓ cuda-memcheck completed successfully${NC}"
        echo "Full report saved to: cuda_memcheck_report.txt"
    else
        echo ""
        echo -e "${RED}✗ cuda-memcheck detected issues${NC}"
        return 1
    fi
}

# Function to run compute-sanitizer (newer CUDA tool)
run_compute_sanitizer() {
    echo ""
    echo "=========================================="
    echo "Running compute-sanitizer (Modern GPU Check)"
    echo "=========================================="

    if ! command -v compute-sanitizer &> /dev/null; then
        echo -e "${YELLOW}Info: compute-sanitizer not found. Skipping.${NC}"
        echo "compute-sanitizer is available in CUDA 11.4+"
        return 1
    fi

    echo "Running memory check..."
    compute-sanitizer --tool memcheck --leak-check full $EXECUTABLE 2>&1 | tee compute_sanitizer_memcheck.txt

    echo ""
    echo "Running race condition check..."
    compute-sanitizer --tool racecheck $EXECUTABLE 2>&1 | tee compute_sanitizer_racecheck.txt

    echo ""
    echo "Running initialization check..."
    compute-sanitizer --tool initcheck $EXECUTABLE 2>&1 | tee compute_sanitizer_initcheck.txt

    echo ""
    echo "Running synchronization check..."
    compute-sanitizer --tool synccheck $EXECUTABLE 2>&1 | tee compute_sanitizer_synccheck.txt

    echo ""
    echo -e "${GREEN}✓ compute-sanitizer checks completed${NC}"
    echo "Reports saved to: compute_sanitizer_*.txt"
}

# Main execution
VALGRIND_RESULT=0
CUDA_MEMCHECK_RESULT=0
COMPUTE_SANITIZER_RESULT=0

run_valgrind || VALGRIND_RESULT=$?
run_cuda_memcheck || CUDA_MEMCHECK_RESULT=$?
run_compute_sanitizer || COMPUTE_SANITIZER_RESULT=$?

# Summary
echo ""
echo "=========================================="
echo "Memory Analysis Summary"
echo "=========================================="

if [ $VALGRIND_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Valgrind: PASSED${NC}"
else
    echo -e "${YELLOW}⚠ Valgrind: SKIPPED or FAILED${NC}"
fi

if [ $CUDA_MEMCHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ cuda-memcheck: PASSED${NC}"
else
    echo -e "${YELLOW}⚠ cuda-memcheck: SKIPPED or FAILED${NC}"
fi

if [ $COMPUTE_SANITIZER_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ compute-sanitizer: PASSED${NC}"
else
    echo -e "${YELLOW}⚠ compute-sanitizer: SKIPPED${NC}"
fi

echo ""
echo "Memory analysis reports generated:"
echo "  - valgrind_report.txt"
echo "  - cuda_memcheck_report.txt"
echo "  - compute_sanitizer_*.txt"
echo ""

# Exit with error if any critical checks failed
if [ $VALGRIND_RESULT -ne 0 ] && [ $VALGRIND_RESULT -ne 1 ]; then
    exit 1
fi
if [ $CUDA_MEMCHECK_RESULT -ne 0 ] && [ $CUDA_MEMCHECK_RESULT -ne 1 ]; then
    exit 1
fi

exit 0
