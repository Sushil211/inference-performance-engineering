#!/bin/bash
echo "🚀 Setting up Inference Performance Engineering Lab..."

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Run 'source venv/bin/activate' to start working."