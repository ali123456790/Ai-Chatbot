#!/bin/bash

# Activate conda environment and run the application
echo "🚀 Starting Historical Document Analyzer..."
echo "📚 Activating conda environment..."

# Initialize conda for this shell session
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# Activate the chatbot environment
conda activate chatbot

echo "✅ Environment activated"
echo "🎯 Starting Streamlit application..."
echo "📝 The app will open in your browser at http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Run the Streamlit app
streamlit run app_gui.py 