import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PDF_FOLDER: str = os.getenv("PDF_FOLDER", "research_papers")
FAISS_PATH: str = os.getenv("FAISS_PATH", "pdf_index.faiss")
METADATA_PATH: str = os.getenv("METADATA_PATH", "metadata.jsonl")

TOP_K: int = int(os.getenv("TOP_K", "10"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
MAX_EMBED_CHARS: int = int(os.getenv("MAX_EMBED_CHARS", "4000"))

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:latest")