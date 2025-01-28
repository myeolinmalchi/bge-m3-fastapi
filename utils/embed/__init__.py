from typing import Literal, Optional

from .onnx import *
from .llama_cpp import *

from utils.logger import _logger

logger = _logger(__name__)

_runtime = None


def init_runtime(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    batch_size: int = 1,
    max_workers: int = 1,
    backend: Literal["onnx", "llama_cpp"] = "onnx",
    device: Literal["cpu", "cuda"] = "cpu",
):
    global _runtime

    if _runtime is None:
        if not model_path:
            raise ValueError("model_path is required")

        match backend:
            case "onnx":
                Embedder = ONNXCpuRuntime if device == "cpu" else ONNXCudaRuntime
                _runtime = Embedder(
                    model_path, tokenizer_path, batch_size, max_workers
                )
            case "llama_cpp":
                if device == "cuda":
                    logger(
                        "'llama_cpp' backend supports cpu only.",
                        level="warning"
                    )
                _runtime = LlamaCppEmbedder(
                    model_path, batch_size=batch_size, max_workers=max_workers
                )

    return _runtime
