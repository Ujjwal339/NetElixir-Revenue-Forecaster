#!/bin/bash

# ==========================================================
# NetElixir Revenue Forecaster
# Run Script
# ==========================================================

echo "==============================================="
echo " NetElixir Revenue Forecaster"
echo "==============================================="
echo

# ----------------------------------------------------------
# Check Virtual Environment
# ----------------------------------------------------------

if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  No Python virtual environment detected."
    echo "Please activate your environment first."
    echo
    echo "Example:"
    echo "conda activate netelixier"
    echo
    exit 1
fi

echo "Using Python:"
python --version
echo

# ----------------------------------------------------------
# Install Dependencies
# ----------------------------------------------------------

echo "Installing required packages..."
pip install -r requirements.txt

echo

# ----------------------------------------------------------
# Create Output Directory
# ----------------------------------------------------------

mkdir -p output

# ----------------------------------------------------------
# Launch Application
# ----------------------------------------------------------

echo "Starting Streamlit application..."
echo

streamlit run app/streamlit_app.py
