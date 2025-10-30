from typing import Iterator, List
import numpy as np
import requests
from .config import EMBED_BATCH_SIZE, EMBED_MODEL, OLLAMA_HOST, MAX_EMBED_CHARS


def embed_text(text: str, retries: int = 2) -> np.ndarray:
    payload = {"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]}
    last_exc = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            vec = np.array(data.get("embedding", []), dtype="float32")
            return vec if vec.ndim == 1 else vec.reshape(-1)
        except Exception as e:
            last_exc = e
    raise last_exc  # type: ignore[misc]


def embed_texts_batched(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> Iterator[np.ndarray]:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for t in batch:
            yield embed_text(t)


def embed_query(query: str) -> np.ndarray:
    return embed_text(query)[None, :]
