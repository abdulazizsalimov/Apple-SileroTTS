#!/bin/bash
#
# Download Silero TTS v5 Russian model for iOS
#
# Usage: ./Scripts/download_model.sh
#

set -e

MODEL_DIR="Models"
MODEL_FILE="v5_ru.pt"
MODEL_URL="https://models.silero.ai/models/tts/ru/v5_ru.pt"

echo "=== Silero TTS Model Downloader ==="
echo ""

# Create Models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo "To re-download, delete the file first."
    exit 0
fi

echo "Downloading Silero v5 Russian TTS model..."
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_DIR/$MODEL_FILE"
echo ""

# Download the model
if command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" --progress-bar
elif command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" --show-progress
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Verify download
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)
    echo ""
    echo "Download complete! Model size: $SIZE"
    echo "Model saved to: $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "Now you need to trace the model for mobile deployment."
    echo "Run: python3 Scripts/trace_model.py"
else
    echo "Error: Download failed!"
    exit 1
fi
