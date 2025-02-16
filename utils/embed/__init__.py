from typing import Literal, Optional
from utils.embed.base import AbsEmbedder
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
) -> AbsEmbedder:
    global _runtime

    if _runtime:
        return _runtime

    if not model_path:
        raise ValueError("model_path is required")

    match (backend, device):
        case ("onnx", "cpu"):
            from .onnx import ONNXCpuRuntime
            tokenizer_path = tokenizer_path if tokenizer_path else "BAAI/bge-m3"
            _runtime = ONNXCpuRuntime(
                model_path,
                tokenizer_path,
                batch_size,
                sessions,
            )

        case ("onnx", "cuda"):
            from .onnx import ONNXCudaRuntime
            tokenizer_path = tokenizer_path if tokenizer_path else "BAAI/bge-m3"
            _runtime = ONNXCudaRuntime(
                model_path,
                tokenizer_path,
                batch_size,
                sessions,
            )

        case ("llama_cpp", "cpu"):
            from .llama_cpp import LlamaCppEmbedder
            _runtime = LlamaCppEmbedder(
                model_path, batch_size=batch_size, max_workers=sessions
            )

        case ("llama_cpp", "cuda"):
            from .llama_cpp import LlamaCppEmbedder
            if device == "cuda":
                logger(
                    "'llama_cpp' backend supports cpu only.", level="warning"
                )
            _runtime = LlamaCppEmbedder(
                model_path, batch_size=batch_size, max_workers=sessions
            )

    return _runtime
