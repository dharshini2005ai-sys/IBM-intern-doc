import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
APP_NAME = os.getenv("APP_NAME", "AI Student Support Assistant")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
FRONTEND_DIR = BASE_DIR / "frontend"
