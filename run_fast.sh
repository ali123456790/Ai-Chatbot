#!/bin/bash

# Fast startup script for development
# This creates an index with 1000 documents in ~60 seconds

echo "🚀 Starting FAST Civil War Document Analyzer..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chatbot

# Check if fast index exists
if [ -f "letter_index_fast.pkl" ]; then
    echo "📚 Fast index found! Starting app directly..."
    streamlit run app_gui.py --server.port 8502 &
    echo "✅ App available at: http://localhost:8502"
else
    echo "📊 Creating fast index (1000 documents)..."
    python chat.py \
        --create-index \
        --xmldir xmlfiles \
        --index-path letter_index_fast.pkl \
        --sample-size 1000 \
        --workers 4 \
        --batch-size 32 \
        --checkpoint-interval 1000 \
        --num-topics 10 \
        --no-hierarchical \
        --no-dynamic \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ Fast index created! Starting app..."
        streamlit run app_gui.py --server.port 8502 &
        echo "🌐 App available at: http://localhost:8502"
    else
        echo "❌ Index creation failed"
        exit 1
    fi
fi

echo "🎯 Fast mode: 1000 documents, no topic modeling, optimized for speed"
echo "📝 For full system: bash run_app.sh" 