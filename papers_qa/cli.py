from typing import Optional
import typer

from .config import FAISS_PATH, METADATA_PATH, TOP_K, OLLAMA_MODEL
from .faiss_store import FaissStore
from .embeddings import embed_query
from .prompt import build_prompt, print_unique_contexts
from .ollama_client import OllamaClient


app = typer.Typer(help="PDF Q&A CLI (FAISS retrieval + Ollama Qwen generation)")


@app.command(help="Index PDFs into FAISS. Use --rebuild to force.")
def index(rebuild: bool = typer.Option(False, help="Recreate index and chunks from scratch")) -> None:
    try:
        FaissStore().build(rebuild=rebuild)
    except Exception as e:
        print(f"Error building index: {e}")


@app.command(help="Ask a question. Starts interactive REPL if no question provided.")
def ask(
    question: Optional[str] = typer.Argument(None, metavar="[QUESTION]", show_default=False),
) -> None:
    try:
        ollama = OllamaClient()
        ollama.ensure_ready()
        # Ensure the generation model is present (auto-pull if missing)
        ollama.ensure_model(OLLAMA_MODEL)
    except RuntimeError as e:
        print(e)
        return

    store = FaissStore()
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
