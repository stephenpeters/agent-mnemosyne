#!/bin/bash
# Run Mnemosyne - Memory, Governance & Orchestration Agent

cd "$(dirname "$0")"

# Set data directory
export MNEMOSYNE_DATA_DIR="/Users/stephenpeters/Library/CloudStorage/Dropbox/Mnemosyne"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Kill any existing process on port 8005
lsof -ti:8005 | xargs kill -9 2>/dev/null || true

echo "Starting Mnemosyne on port 8005..."
uvicorn main:app --reload --port 8005
