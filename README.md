# Papers-QA

A command-line tool to answer questions about your PDF research papers using FAISS retrieval and Qwen (via Ollama).

## Features
- Token-based PDF chunking with section awareness
- Local FAISS vector index for fast retrieval
- Open-source embeddings served by Ollama (default: `nomic-embed-text`)
- Qwen generation via Ollama (auto-start and model pull)
- Context dedup to avoid duplicate source lines

## Requirements
- Python >= 3.10
- FAISS (mandatory for index and ask)
- Ollama installed (`ollama`) and models available:
  - Embeddings model (default `nomic-embed-text`)
  - Generation model (default `qwen3:latest`)

## Installation
Choose one:

- pipx (recommended)
```
pipx install .


- Poetry 
```
poetry install
poetry run papers-qa --help
# force a specific Python
poetry install
```

- venv + pip
```
python3 -m venv .venv
. .venv/bin/activate
pip install .
papers-qa --help
```

## Usage
1) Interactive mode (recommended start):
```
papers-qa ask
```
2) Ask a one‑off question:
```
papers-qa ask "describe the ProSEA architecture""
```
3) Optional — index more documents (after adding PDFs to `research_papers/` or setting `PDF_FOLDER`), then rebuild:
```
papers-qa index --rebuild
```

## Examples
### Question related to the documents
`papers-qa ask "describe the ProSEA architecture"`

```bash
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
`papers-qa ask "what's the meaning of life"`

```bash
...
I don't know have enought information from your knowledge base
```


## Usage notes
- The CLI auto-starts `ollama serve` if not running and pulls the embedding and generation models as needed.

## Tuning
No retrieval threshold is applied in this version; FAISS top‑k is used directly.

## Add documents (optional)
- Drop PDFs into `research_papers/` (or your `PDF_FOLDER`).
- Rebuild index to include them:
```
papers-qa index --rebuild
```
- Storage: FAISS at `pdf_index.faiss`, metadata at `metadata.jsonl` (paths configurable). Both are regenerated on `--rebuild`.

## Project structure
- `papers_qa/`
  - `cli.py`: Typer CLI entry (`index`, `ask`).
  - `config.py`: environment-driven config (paths, models, thresholds).
  - `chunking.py`: PDF parsing and token chunking with section hints.
  - `embeddings.py`: Ollama-based embeddings (batch + query).
  - `faiss_store.py`: `FaissStore` (build/load/retrieve for FAISS + metadata).
  - `prompt.py`: prompt template, builders, unique context printing.
  - `ollama_client.py`: `OllamaClient` (ensure daemon/model, call/stream).
- `research_papers/`: default PDFs folder.
- `pdf_index.faiss`, `metadata.jsonl`: index and aligned metadata.
- `pyproject.toml`, `README.md`.

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
    "Answer (ONLY from context, or say 'I don't know have enought information from your knowledge base'):"
```

## Improvements
- Meta-questions doesn't work. For example: “What do all PDFs talk about?” won’t match vector chunks reliably. We could add a small router agent to detect meta vs. content-specific queries. For meta, use per-paper summaries/keywords to answer without chunk retrieval.
- Retrieval threshold: add a token‑overlap (e.g., Jaccard) threshold to skip LLM calls when no retrieved chunk is sufficiently related, saving latency/cost.
- Summary indexing: store per-paper summaries/keywords in a separate index for fast meta answers.
- Diversified retrieval across papers: current TOP_K may cluster chunks from one paper. Use a two-stage approach: retrieve a larger candidate set (e.g., 5×TOP_K), then take a small per-file top_k (e.g., 2–3) to ensure multi-paper coverage.
- A lot of tuning for the chunk size, model to use, prompt, etc...
