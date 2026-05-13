import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY tidak ditemukan")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

TICKET_FILE = DATA_DIR / "Testing Data - Tiket Support.csv"
# DATASET_PATH = DATA_DIR / "Testing Data - Tiket Support.csv"

# Model embedding multilingual
EMBEDDING_MODEL      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME      = "tiket_cloudkilat"
LLM_MODEL            = "gemini-2.5-flash"
TEMPERATURE          = 0.3

# HYBRID RAG PARAMS
USE_HYBRID = True

# weight kombinasi
WEIGHT_SEMANTIC = 0.8
WEIGHT_BM25     = 0.2
RETRIEVER_K = 5
BM25_K = 3

# threshold relevansi final
SIMILARITY_THRESHOLD = 0.45