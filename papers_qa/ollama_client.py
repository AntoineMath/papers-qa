from __future__ import annotations
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests

from .config import OLLAMA_HOST, OLLAMA_MODEL


@dataclass(slots=True)
class OllamaClient:
    host: str = OLLAMA_HOST
    model: str = OLLAMA_MODEL
    timeout_s: int = 600

    # TODO: check env are set

    def call(self, prompt: str) -> str:
        """Perform a blocking call to Ollama."""
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream response lines from Ollama."""
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        try:
            with requests.post(url, json=payload, timeout=self.timeout_s, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield obj.get("response", "")
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama streaming failed: {e}") from e

    def ensure_ready(self) -> None:
        """Ensure Ollama daemon and model are available."""
        if self._is_up():
            self._ensure_model_present()
            return
        self._start_daemon()
        deadline = time.time() + 60
        while time.time() < deadline:
            if self._is_up():
                self._ensure_model_present()
                return
            time.sleep(0.5)
        raise RuntimeError("Failed to start Ollama daemon.")

    def ensure_model(self, model: str) -> None:
        """Ensure a specific model is available locally (pull if missing)."""
        self._ensure_model_present(model)

    def _is_up(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _start_daemon(self) -> None:
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not installed or not on PATH.")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _ensure_model_present(self, timeout_s: int = 1800) -> None:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10)
            tags.raise_for_status()
            models = tags.json().get("models", [])
            if any(m.get("name") == self.model for m in models):
                return
        except requests.RequestException:
            pass
        resp = requests.post(
            f"{self.host}/api/pull",
            json={"model": self.model, "stream": False},
            timeout=timeout_s,
        )
        resp.raise_for_status()