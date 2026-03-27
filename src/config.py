import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # API Settings
    PORT = int(os.getenv("PORT", 8000))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
    # Model Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models/artifacts/best_model.pkl"))
    PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", str(BASE_DIR / "models/artifacts/preprocessor.pkl"))
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Security
    API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    API_KEY = os.getenv("API_KEY", "")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 100))
    
    # Data Paths
    DATA_PATH = BASE_DIR / "data/raw/Student_Performance.csv"
    
    @classmethod
    def display(cls):
        """Display configuration (hide sensitive data)"""
        print("="*50)
        print("Configuration:")
        print(f"  Environment: {cls.ENVIRONMENT}")
        print(f"  Port: {cls.PORT}")
        print(f"  Model Path: {cls.MODEL_PATH}")
        print(f"  CORS Origins: {cls.CORS_ORIGINS}")
        print(f"  API Key Enabled: {cls.API_KEY_ENABLED}")
        print("="*50)
    
config = Config()
