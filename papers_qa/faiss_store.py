import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import faiss
from tqdm import tqdm

from .config import PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_PATH, METADATA_PATH, TOP_K, EMBED_MODEL
from .chunking import pdf_to_token_chunks
from .embeddings import embed_texts_batched, embed_text
from .ollama_client import OllamaClient


@dataclass(slots=True)
class FaissStore:
    index_path: str = FAISS_PATH
    metadata_path: str = METADATA_PATH
    _index: Optional[faiss.Index] = field(default=None, init=False)
    _metadata: Optional[List[Tuple[str, str, str]]] = field(default=None, init=False)

    def build(self, rebuild: bool = False) -> None:
        if (not rebuild) and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Index exists. Use --rebuild to recreate.")
            return

        items = pdf_to_token_chunks(PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
        if not items:
            raise RuntimeError("No PDF chunks found.")

        # Ensure Ollama is ready and the embedding model is present
        try:
            oc = OllamaClient()
            oc.ensure_ready()
            oc.ensure_model(EMBED_MODEL)
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings not available ({e}). Ensure Ollama is running and model '{EMBED_MODEL}' exists.")

        index: Optional[faiss.IndexFlatL2] = None
        buffer: List[np.ndarray] = []
        emb_dim: Optional[int] = None

        if rebuild:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)

        total = len(items)
        with open(self.metadata_path, "w", encoding="utf-8") as meta_out:
            pbar = tqdm(total=total, desc="Indexing", unit="chunk")
            for file_name, section, chunk_text in items:
                try:
                    vec = embed_text(chunk_text)
                except Exception:
                    # Skip problematic chunk
                    pbar.update(1)
                    continue

                if emb_dim is None:
                    emb_dim = vec.shape[0]
                    index = faiss.IndexFlatL2(emb_dim)

                buffer.append(vec)
                meta_out.write(json.dumps({"file": file_name, "section": section, "chunk": chunk_text}) + "\n")
                pbar.update(1)

                if len(buffer) >= 256:
                    mat = np.vstack(buffer).astype("float32")
                    index.add(mat)
                    buffer.clear()

            if buffer:
                mat = np.vstack(buffer).astype("float32")
                index.add(mat)
            pbar.close()

        if index is None:
            raise RuntimeError("FAISS index was not initialized")
        faiss.write_index(index, self.index_path)
        self._index = index
        # don't load metadata into memory here; leave lazy
        print(f"Built index with {index.ntotal} vectors. Metadata at {self.metadata_path}.")

    def load_index(self) -> faiss.Index:
        if self._index is not None:
            return self._index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self._index = faiss.read_index(self.index_path)
        return self._index

    def load_metadata(self) -> List[Tuple[str, str, str]]:
        if self._metadata is not None:
            return self._metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        items: List[Tuple[str, str, str]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                items.append((obj["file"], obj["section"], obj["chunk"]))
        self._metadata = items
        return items

    def retrieve(self, query_vec: np.ndarray, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        index = self.load_index()
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        dists, idxs = index.search(query_vec, top_k)
        return dists[0], idxs[0]
