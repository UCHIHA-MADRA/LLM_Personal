"""
LLM Engine — The core inference engine.
Loads GGUF model files directly using llama-cpp-python.
No servers, no APIs, no external dependencies.
Your Python process IS the LLM.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Try importing llama_cpp — if not installed, we provide helpful instructions
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class LLMEngine:
    """
    Core LLM inference engine.
    Loads a GGUF model file and runs generation entirely on your hardware.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.model_name: str = "None"
        self._is_loaded = False

    @staticmethod
    def check_dependencies() -> bool:
        """Check if llama-cpp-python is installed."""
        if not LLAMA_CPP_AVAILABLE:
            print("\n" + "=" * 60)
            print("❌  llama-cpp-python is NOT installed.")
            print("=" * 60)
            print("\nInstall it with:")
            print()
            print("  CPU only:")
            print("    pip install llama-cpp-python")
            print()
            print("  NVIDIA GPU (recommended for speed):")
            print('    set CMAKE_ARGS="-DGGML_CUDA=on"')
            print("    pip install llama-cpp-python --force-reinstall --no-cache-dir")
            print()
            print("  For pre-built wheels (easier):")
            print("    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
            print("=" * 60 + "\n")
            return False
        return True

    def load(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        verbose: bool = False,
        chat_format: Optional[str] = None,
    ) -> bool:
        """
        Load a GGUF model file from disk.

        Args:
            model_path: Absolute path to the .gguf file
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
            n_ctx: Context window size in tokens
            verbose: Print model loading details
            chat_format: Chat template format (e.g., 'chatml', 'llama-3', 'mistral-instruct')
        """
        if not self.check_dependencies():
            return False

        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"Model file not found: {model_path}")
            print(f"\n❌ Model file not found: {model_path}")
            return False

        if not str(model_file).endswith(".gguf"):
            logger.error(f"Not a GGUF file: {model_path}")
            print(f"\n❌ Not a GGUF file: {model_path}")
            return False

        # Unload previous model if any
        self.unload()

        try:
            size_gb = model_file.stat().st_size / (1024**3)
            print(f"\n🔄 Loading model: {model_file.name} ({size_gb:.1f} GB)")
            print(f"   GPU layers: {'ALL' if n_gpu_layers == -1 else n_gpu_layers}")
            print(f"   Context: {n_ctx} tokens")

            start = time.time()

            kwargs = {
                "model_path": str(model_file),
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "verbose": verbose,
                "n_threads": None,  # auto-detect
            }

            if chat_format:
                kwargs["chat_format"] = chat_format

            self.model = Llama(**kwargs)
            self.model_path = str(model_file)
            self.model_name = model_file.stem
            self._is_loaded = True

            elapsed = time.time() - start
            print(f"✅ Model loaded in {elapsed:.1f}s")
            print(f"   Model: {self.model_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"\n❌ Failed to load model: {e}")
            self._is_loaded = False
            return False

    def unload(self):
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            self.model_path = None
            self.model_name = "None"
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            # On Windows, we can try to empty the working set to free up RAM immediately
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
                except Exception:
                    pass
                    
            logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt: The input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Randomness (0.0 = deterministic, 1.0+ = creative)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            repeat_penalty: Penalize repeated tokens
            stop: Stop sequences
            stream: If True, yields tokens as they're generated

        Returns:
            If stream=False: complete response string
            If stream=True: generator yielding token strings
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        if stream:
            return self._generate_stream(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            )
        else:
            return self._generate_complete(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            )

    def _generate_complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[List[str]],
    ) -> str:
        """Generate a complete response at once."""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[List[str]],
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time."""
        stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
            stream=True,
        )

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stream: bool = False,
    ):
        """
        Multi-turn chat using the model's chat template.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            max_tokens: Maximum tokens in response
            temperature: Randomness control
            stream: If True, yields tokens as generated

        Returns:
            If stream=False: complete response string
            If stream=True: generator yielding token strings
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        if stream:
            return self._chat_stream(messages, max_tokens, temperature, top_p, repeat_penalty)
        else:
            return self._chat_complete(messages, max_tokens, temperature, top_p, repeat_penalty)

    def _chat_complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
    ) -> str:
        """Chat completion — returns full response."""
        try:
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Fallback: manually format prompt if chat template fails
            logger.warning(f"Chat template failed ({e}), falling back to manual prompt")
            prompt = self._format_chat_prompt(messages)
            return self._generate_complete(
                prompt, max_tokens, temperature, top_p, 40, repeat_penalty, None
            )

    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
    ) -> Generator[str, None, None]:
        """Chat completion — streams tokens."""
        try:
            stream = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        except Exception as e:
            # Fallback: manually format prompt if chat template fails
            logger.warning(f"Chat template streaming failed ({e}), falling back")
            prompt = self._format_chat_prompt(messages)
            yield from self._generate_stream(
                prompt, max_tokens, temperature, top_p, 40, repeat_penalty, None
            )

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Fallback: manually format messages into a prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant: ")
        return "\n".join(parts)

    def get_info(self) -> Dict[str, Any]:
        """Get info about the currently loaded model."""
        if not self.is_loaded:
            return {"loaded": False, "name": "None"}

        info = {
            "loaded": True,
            "name": self.model_name,
            "path": self.model_path,
        }

        try:
            file_size = Path(self.model_path).stat().st_size / (1024**3)
            info["size_gb"] = round(file_size, 2)
        except Exception:
            pass

        try:
            info["n_ctx"] = self.model.n_ctx()
        except Exception:
            pass

        return info


# ─── Module-level singleton ───────────────────────────────────
_engine = None

def get_engine() -> LLMEngine:
    """Get the global LLM engine instance."""
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine
