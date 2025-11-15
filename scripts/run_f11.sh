#!/bin/bash

# Run F11 benchmark
# Requires: GROQ_API_KEY environment variable to be set
# Usage: export GROQ_API_KEY="your-key-here" && ./run_f11.sh

if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY environment variable not set"
    echo "Usage: export GROQ_API_KEY='your-key-here' && ./run_f11.sh"
    exit 1
fi

python3 run_f11_benchmark.py
