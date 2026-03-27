# Add this at the top of your app.py
import sys
sys.path.append('.')
from src.config import config

# In your app initialization, use config
print(f"Starting API in {config.ENVIRONMENT} mode on port {config.PORT}")

# Update model paths
MODEL_PATH = config.MODEL_PATH
PREPROCESSOR_PATH = config.PREPROCESSOR_PATH

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS != ['*'] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
