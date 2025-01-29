from typing import Literal, Optional
from utils.logger import _logger

logger = _logger(__name__)
_runtime = None


def init_runtime(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    batch_size: int = 1,
    sessions: int = 1,
    backend: Literal["onnx", "llama_cpp"] = "onnx",
    device: Literal["cpu", "cuda"] = "cpu",
):
    global _runtime

    if _runtime is None:
        if not model_path:
            raise ValueError("model_path is required")

        match backend:
            case "onnx":
                from .onnx import ONNXCpuRuntime, ONNXCudaRuntime
                Embedder = ONNXCpuRuntime if device == "cpu" else ONNXCudaRuntime
                tokenizer_path = tokenizer_path if tokenizer_path else "BAAI/bge-m3"
                _runtime = Embedder(
                    model_path,
                    tokenizer_path,
                    batch_size,
                    sessions,
                )
            case "llama_cpp":
                from .llama_cpp import LlamaCppEmbedder
                if device == "cuda":
                    logger(
                        "'llama_cpp' backend supports cpu only.",
                        level="warning"
                    )
                _runtime = LlamaCppEmbedder(
                    model_path, batch_size=batch_size, max_workers=sessions
                )

    return _runtime
