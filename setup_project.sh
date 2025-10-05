#!/bin/bash

echo "Setting up Exoplanet ML project structure..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/plots
mkdir -p src/ui
mkdir -p tests
mkdir -p scripts
mkdir -p .github/workflows
mkdir -p notebooks

touch src/__init__.py
touch src/ui/__init__.py
touch tests/__init__.py

echo "Directory structure created!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Download data: python scripts/download_data.py"
echo "3. Train model: python src/train.py"
echo "4. Run API: uvicorn src.api:app --reload"
echo "5. Run UI: streamlit run src/ui/streamlit_app.py"
