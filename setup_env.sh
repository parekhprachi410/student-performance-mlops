#!/bin/bash

echo "Setting up environment variables..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << 'ENV'
# API Configuration
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=debug

# Model Paths
MODEL_PATH=models/artifacts/best_model.pkl
PREPROCESSOR_PATH=models/artifacts/preprocessor.pkl

# MLflow
MLFLOW_TRACKING_URI=mlruns

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:8501,http://localhost:8000

# Security (disabled by default for development)
API_KEY_ENABLED=false
API_KEY=

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
ENV
    echo "✅ .env file created"
else
    echo "⚠️ .env file already exists"
fi

# Display current configuration
echo ""
echo "Current configuration:"
echo "PORT: ${PORT:-8000}"
echo "ENVIRONMENT: ${ENVIRONMENT:-development}"
echo "MODEL_PATH: ${MODEL_PATH:-models/artifacts/best_model.pkl}"

echo ""
echo "To load variables, run: source .env"
