import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMADB_DIR = DATA_DIR / "chromadb"

# Load environment variables from .env file FIRST
rag_env = BASE_DIR / ".env"
if rag_env.exists():
    load_dotenv(dotenv_path=rag_env)
else:
    load_dotenv()

# Configure HuggingFace environment variables & HF_TOKEN authentication
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"


EVAL_DATA_DIR = BASE_DIR / "eval_data"
EVAL_DATASET_CSV = EVAL_DATA_DIR / "eval_dataset.csv"
EVAL_DATASET_JSON = EVAL_DATA_DIR / "eval_dataset.json"
EVAL_REPORTS_DIR = EVAL_DATA_DIR / "eval_reports"

# Ensure necessary directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Vector Store & Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# LLM Configuration
LLM_TYPE = os.getenv("LLM_TYPE", "openai").lower()  # options: ollama, openai, gemini, anthropic, mock
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# HF Token Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Collection Name in ChromaDB
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
