#!/bin/bash
# Dataset download instructions and automation
# Run: bash scripts/download_data.sh

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "=== AAPR-Speech Dataset Download ==="
echo ""

# --- Crema-D ---
echo "1. Crema-D Dataset"
echo "   Source: https://www.kaggle.com/datasets/ejlok1/cremad"
echo "   Download via Kaggle CLI:"
echo "   kaggle datasets download -d ejlok1/cremad -p $DATA_DIR/cremad --unzip"
echo ""

if command -v kaggle &> /dev/null; then
    read -p "Download Crema-D now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kaggle datasets download -d ejlok1/cremad -p "$DATA_DIR/cremad" --unzip
        echo "Crema-D downloaded successfully."
    fi
else
    echo "   (kaggle CLI not found. Install with: pip install kaggle)"
fi

echo ""

# --- MDER-MA ---
echo "2. MDER-MA Dataset"
echo "   Source: Mendeley Data"
echo "   Manual download required: https://data.mendeley.com/"
echo "   Extract to: $DATA_DIR/mderma/"
echo ""

# --- TAME ---
echo "3. TAME Dataset"
echo "   Source: PhysioNet"
echo "   Requires PhysioNet credentialed access"
echo "   Download instructions: https://physionet.org/"
echo "   Extract to: $DATA_DIR/tame/"
echo ""

echo "=== Done ==="
echo "After downloading, verify structure:"
echo "  data/raw/cremad/AudioWAV/*.wav"
echo "  data/raw/cremad/VideoDemographics.csv"
echo "  data/raw/mderma/*.wav (or subdirectories)"
echo "  data/raw/tame/*.wav + metadata.csv"
