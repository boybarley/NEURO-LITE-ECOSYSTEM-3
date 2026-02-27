"""
Neuro-Lite LLM Engine
─────────────────────
Singleton wrapper around llama-cpp-python.
Thread-safe with single inference lock.
Optimized for 4GB RAM / i3 CPU.
"""

import os
import threading
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Generator, Any

logger = logging.getLogger("neurolite.llm")

BASE_DIR = Path(__file__).resolve().parent.parent


class LLMEngine:
    """
    Thread-safe singleton LLM engine.
    Ensures only one inference runs at a time.
    Supports streaming token generation.
    """

    _instance: Optional["LLMEngine"] = None
    _creation_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._creation_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._ready = False
            return cls._instance

    def __init__(self):
        if self._ready:
            return
        self._llm = None
        self._inference_lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._load_error: Optional[str] = None
        self._ready = True

    def load_model(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = 2,
        n_batch: int = 64,
    ) -> bool:
        """
        Load GGUF model. Returns True on success.
        """
        if model_path is None:
            model_path = os.environ.get(
                "MODEL_PATH",
                str(BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
            )

        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = BASE_DIR / resolved

        if not resolved.exists():
            self._load_error = f"Model file not found: {resolved}"
            logger.error(self._load_error)
            return False

        with self._inference_lock:
            try:
                # Free previous model if switching
                if self._llm is not None:
                    del self._llm
                    self._llm = None
                    import gc
                    gc.collect()

                logger.info(
                    "Loading model: %s (n_ctx=%d, threads=%d, batch=%d)",
                    resolved.name, n_ctx, n_threads, n_batch,
                )
                start = time.monotonic()

                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=str(resolved),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    n_threads_batch=n_threads,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                    chat_format="chatml",
                )

                elapsed = time.monotonic() - start
                self._model_path = str(resolved)
                self._load_error = None
                logger.info("Model loaded in %.1fs: %s", elapsed, resolved.name)
                return True

            except Exception as e:
                self._load_error = str(e)
                logger.error("Failed to load model: %s", e)
                self._llm = None
                return False

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_name(self) -> Optional[str]:
        if self._model_path:
            return Path(self._model_path).name
        return None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming chat completion.
        Blocks until inference_lock is available.
        Yields individual tokens as strings.
        """
        if not self.is_loaded:
            yield "[Error: Model not loaded]"
            return

        acquired = self._inference_lock.acquire(timeout=120)
        if not acquired:
            yield "[Error: Inference busy — please wait]"
            return

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop or [],
                stream=True,
            )

            for chunk in response:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token

        except Exception as e:
            logger.error("Inference error: %s", e, exc_info=True)
            yield f"\n[Inference error: {e}]"
        finally:
            self._inference_lock.release()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> str:
        """Non-streaming completion. Blocks until done."""
        tokens = []
        for token in self.generate_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        ):
            tokens.append(token)
        return "".join(tokens)

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._inference_lock:
            if self._llm is not None:
                del self._llm
                self._llm = None
                import gc
                gc.collect()
                logger.info("Model unloaded.")
            self._model_path = None
