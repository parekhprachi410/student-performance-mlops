FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_PATH=models/artifacts/best_model.pkl
ENV PREPROCESSOR_PATH=models/artifacts/preprocessor.pkl

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p models/artifacts data/raw

# Expose port
EXPOSE ${PORT}

# Run the API
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}
