import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Resolve base dir for relative fallbacks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths and dataset (env vars prefixed with PAPERS_QA_)
PDF_FOLDER: str = os.getenv("PAPERS_QA_PDF_FOLDER", "research_papers")
METADATA_PATH: str = os.getenv(
    "PAPERS_QA_METADATA_PATH",
    os.path.join(BASE_DIR, "..", "metadata.jsonl"),
)
FAISS_PATH: str = os.getenv(
    "PAPERS_QA_FAISS_PATH",
    os.path.join(BASE_DIR, "..", "pdf_index.faiss"),
)

# Chunking/embeddings
TOP_K: int = int(os.getenv("PAPERS_QA_TOP_K", "10"))
CHUNK_SIZE: int = int(os.getenv("PAPERS_QA_CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("PAPERS_QA_CHUNK_OVERLAP", "50"))
EMBED_BATCH_SIZE: int = int(os.getenv("PAPERS_QA_EMBED_BATCH_SIZE", "64"))
EMBED_MODEL: str = os.getenv("PAPERS_QA_EMBED_MODEL", "nomic-embed-text")
MAX_EMBED_CHARS: int = int(os.getenv("PAPERS_QA_MAX_EMBED_CHARS", "4000"))

# Ollama
OLLAMA_HOST: str = os.getenv("PAPERS_QA_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("PAPERS_QA_OLLAMA_MODEL", "qwen3:latest")