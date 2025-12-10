#!/bin/bash

# Script to set up the CRAG conda environment

echo "========================================="
echo "Setting up CRAG conda environment"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Environment created successfully!"
    echo "========================================="
    echo ""
    echo "To activate the environment, run:"
    echo "    conda activate crag"
    echo ""
    echo "To deactivate the environment, run:"
    echo "    conda deactivate"
    echo ""
    echo "To verify the installation, run:"
    echo "    conda activate crag"
    echo "    python verify_installation.py"
else
    echo ""
    echo "Error: Failed to create conda environment"
    echo "If the environment already exists, you can update it with:"
    echo "    conda env update -f environment.yml"
    exit 1
fi
