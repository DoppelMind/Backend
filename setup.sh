#!/bin/bash
set -e

echo "Setting up DoppelMind Backend..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please add your MISTRAL_API_KEY before starting."
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. Edit .env and set your MISTRAL_API_KEY"
echo "  2. Run: source venv/bin/activate && uvicorn main:app --reload --port 8000"
