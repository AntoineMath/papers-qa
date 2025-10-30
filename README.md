# Papers-QA

A command-line tool to answer questions about your PDF research papers using FAISS retrieval and Qwen (via Ollama) locally.

## Features
- Token-based PDF chunking with pdf section awareness
- Local FAISS vector index for fast retrieval
- Open-source embeddings served by Ollama (default: `nomic-embed-text`)
- Qwen generation via Ollama (auto-start and model pull)

## Requirements
- Python >= 3.10
- Ollama installed (`ollama`) and models available:
  - Embeddings model (default `nomic-embed-text`)
  - Generation model (default `qwen3:latest`)

## Installation
Choose one:

### pipx (recommended)
```
pipx install .

```

### Poetry 
```
poetry install
poetry run papers-qa --help
```

### venv + pip
```
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## Usage
```
papers-qa --help
```  

starts Interactive mode  
```
papers-qa ask
```  

Ask a one question  
```
papers-qa ask "describe the ProSEA architecture"

```

Optional — index more documents
```
papers-qa index --rebuild
```

## Examples
### Question related to the documents
```
papers-qa ask "describe the ProSEA architecture"
```

```
Context used:
- 2510.07423v1.pdf | 1 Introduction
- 2510.07423v1.pdf | 2.2 Human-AI Collaboration and Interaction
- 2510.07423v1.pdf | 4 Experiments
- 2510.07423v1.pdf | Introduction
- 2510.07423v1.pdf | 3.2 Expert Agents and the Exploration
- 2510.07423v1.pdf | 5 Conclusion
- 2510.08149v1.pdf | 7 Conclusion
The ProSEA architecture is a **multi-agent framework** designed for effective problem-solving through **iterative exploration** without requiring human guidance or domain-specific training. Key components and features include:  

1. **Two-Dimensional Exploration**: Combines **rich feedback mechanisms** with **iterative exploration** to navigate complex tasks. This likely involves simultaneous exploration across different dimensions (e.g., strategy, data sources, or problem-solving approaches).  

2. **Expert Agents**: These agents adaptively refine their approaches based on **discovered constraints** and **failures**. They dynamically adjust strategies to overcome obstacles, enabling the system to find viable solutions without predetermined paths.  

3. **Autonomous Operation**: The framework operates fully autonomously, outperforming traditional systems like RAG and ReAct while achieving performance comparable to human-intervention-dependent systems like DANA.  

4. **Human-in-the-Loop Collaboration**: While experiments focus on autonomy, ProSEA’s architecture inherently supports integration with human expertise, enhancing real-world deployment by complementing automated reasoning with human input.  

5. **Adaptive Refinement**: Systematic exploration with adaptive refinement replaces manual planning, allowing ProSEA to iteratively improve solutions through feedback and constraint analysis.  

This structure emphasizes flexibility, adaptability, and collaboration between automated agents and human expertise.
```

### Question not related to the documents
```
papers-qa ask "what's the meaning of life"
```

`
...
I don't know have enought information from your knowledge base
`


## Usage notes
- The CLI auto-starts `ollama serve` if not running and pulls the embedding and generation models as needed.
- Embeddings and Qwen3 responses run locally via Ollama; outputs, latency, and quality may vary across machines (hardware, model versions).

By default, the application uses an already created (packaged) database and metadata. You can override them or create new ones with the `index` command:
- Via env vars: `PAPERS_QA_FAISS_PATH` and `PAPERS_QA_METADATA_PATH`
- Per-command flags:
  - index: `--db /path/to/pdf_index.faiss --meta /path/to/metadata.jsonl`
  - ask: `--db /path/to/pdf_index.faiss --meta /path/to/metadata.jsonl`

## Configuration
Optional: you can tweek the behaviour of the tool with those env variables.
```
PAPERS_QA_PDF_FOLDER=research_papers
PAPERS_QA_FAISS_PATH=pdf_index.faiss
PAPERS_QA_METADATA_PATH=metadata.jsonl
PAPERS_QA_OLLAMA_HOST=http://localhost:11434
PAPERS_QA_OLLAMA_MODEL=qwen3:latest
PAPERS_QA_EMBED_MODEL=nomic-embed-text
PAPERS_QA_TOP_K=5
PAPERS_QA_CHUNK_SIZE=500
PAPERS_QA_CHUNK_OVERLAP=50
PAPERS_QA_EMBED_BATCH_SIZE=64
PAPERS_QA_MAX_EMBED_CHARS=4000
```

## Project structure
- `papers_qa/`
  - `cli.py`: Typer CLI entry (`index`, `ask`).
  - `config.py`: environment-driven config (PAPERS_QA_* envs, paths, models).
  - `chunking.py`: PDF parsing and token chunking with section hints.
  - `embeddings.py`: Ollama-based embeddings (batch + query).
  - `faiss_store.py`: `FaissStore` (build/load/retrieve for FAISS + metadata; loads packaged defaults if user paths absent).
  - `prompt.py`: prompt template, builders, unique context printing.
  - `ollama_client.py`: `OllamaClient` (ensure daemon, pull models, call/stream).
  - `metadata.jsonl` (packaged default metadata, optional).
  - `pdf_index.faiss` (packaged default FAISS index, optional).
- PDFs folder: set via `PAPERS_QA_PDF_FOLDER` (default `research_papers/`).
- Root: `pyproject.toml`, `README.md`.

## Technology overview
- FAISS: stores embedding vectors for chunks; enables nearest-neighbor search.
- metadata.jsonl: one line per vector; line i holds `{file, section, chunk}` for FAISS vector i.
- Embeddings via Ollama: calls `/api/embeddings` using `EMBED_MODEL` per text (default: `nomic-embed-text`).
- Qwen via Ollama: generation through `/api/generate` (streaming supported).
- Prompt:
```
  "You are a helpful research assistant.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Answer ONLY using the provided context below. Do NOT use general knowledge.\n"
    "2. If information is not explicitly in the context, you MUST say 'I don't know'.\n"
    "3. Do not make inferences or assumptions beyond what is stated.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (ONLY from context, or say 'I don't have enought information from your knowledge base'):"
```

## Improvements
### Meta-questions doesn't work
For example: “What do all PDFs talk about?” won’t match vector chunks reliably. We could add a small router agent to detect meta vs. content-specific queries. For meta, use per-paper summaries/keywords to answer without chunk retrieval.
### Retrieval threshold
add a token‑overlap (e.g., Jaccard) threshold to skip LLM calls when no retrieved chunk is sufficiently related, saving latency/cost.
### Avoid focusing on one paper
Diversified retrieval across papers: current TOP_K may cluster chunks from one paper. Use a two-stage approach: retrieve a larger candidate set (e.g., 5×TOP_K), then take a small per-file top_k (e.g., 2–3) to ensure multi-paper coverage.
### Other
- A lot of tuning for the chunk size, model to use, prompt, etc...
- Summary indexing: store per-paper summaries/keywords in a separate index for fast meta answers.
- Proper testing. Try build reproductible tests
