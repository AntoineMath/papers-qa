from typing import Optional
import os
import typer

from .config import TOP_K, OLLAMA_MODEL, EMBED_MODEL, FAISS_PATH, METADATA_PATH
from .faiss_store import FaissStore
from .embeddings import embed_query
from .prompt import build_prompt, print_unique_contexts
from .ollama_client import OllamaClient


app = typer.Typer(help="PDF Q&A CLI (FAISS retrieval + Ollama Qwen generation)")


@app.command(help="Index PDFs into FAISS. Use --rebuild to force.")
def index(
    rebuild: bool = typer.Option(False, help="Recreate index and chunks from scratch"),
    db: Optional[str] = typer.Option(None, "--db", help="output path to FAISS index.faiss file (overrides env)"),
    data: Optional[str] = typer.Option(None, "--meta", help="output path to metadata.jsonl (overrides env)"),
    input: Optional[str] = typer.Option(None, "--input", help="path to input PDFs folder (overrides env)"),
) -> None:
    try:
        # Resolve paths (CLI flag takes precedence over env/config)
        in_path = input or os.getenv("PAPERS_QA_PDF_FOLDER") or None
        db_path = db or os.getenv("PAPERS_QA_FAISS_PATH") or None
        data_path = data or os.getenv("PAPERS_QA_METADATA_PATH") or None

        # Validate input folder
        if not in_path or not os.path.isdir(in_path):
            print("Input folder not found. Pass --input or set PAPERS_QA_PDF_FOLDER to a valid directory.")
            return

        # Ensure output directories exist (create if needed)
        if not db_path:
            print("Missing output index path. Pass --db or set PAPERS_QA_FAISS_PATH.")
            return
        if not data_path:
            print("Missing output metadata path. Pass --data or set PAPERS_QA_METADATA_PATH.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(data_path)) or ".", exist_ok=True)

        store = FaissStore(index_path=db_path, metadata_path=data_path)
        store.build(rebuild=rebuild, folder_path=in_path)
    except Exception as e:
        print(f"Error building index: {e}")


@app.command(help="Ask a question. Starts interactive REPL if no question provided.")
def ask(
    question: Optional[str] = typer.Argument(None, metavar="[QUESTION]", show_default=False),
    db: Optional[str] = typer.Option(None, "--db", help="input path to FAISS .faiss file (overrides env)"),
    data: Optional[str] = typer.Option(None, "--meta", help="input path to metadata.jsonl (overrides env)"),
) -> None:
    try:
        ollama = OllamaClient()
        ollama.ensure_ready()
        print(f"checking model '{EMBED_MODEL}' (download if missing)...", flush=True)
        ollama.ensure_model(EMBED_MODEL)
        print(f"checking model '{OLLAMA_MODEL}' (download if missing)...", flush=True)
        ollama.ensure_model(OLLAMA_MODEL)
    except RuntimeError as e:
        print(e)
        return

    # Allow overriding input index/metadata via CLI
    store = FaissStore(index_path=db or FAISS_PATH, metadata_path=data or METADATA_PATH)
    metadata = store.load_metadata()

    def run_query(q: str) -> None:
        qvec = embed_query(q)
        _, idxx = store.retrieve(qvec, TOP_K)
        candidates = [(metadata[i][0], metadata[i][1], metadata[i][2]) for i in idxx if 0 <= i < len(metadata)]
        contexts = candidates
        if not contexts:
            print("No relevant context found.")
            return
        print_unique_contexts(contexts)
        prompt = build_prompt(q, contexts)
        try:
            print("thinking...\n", flush=True)
            for piece in ollama.stream(prompt):
                print(piece, end="", flush=True)
            print()
        except RuntimeError as e:
            print(e)

    if question:
        run_query(question.strip())
        return

    print("RAG CLI (Qwen via Ollama). Type 'exit' to quit.")
    while True:
        q = input("?> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        run_query(q)


if __name__ == "__main__":
    app()
