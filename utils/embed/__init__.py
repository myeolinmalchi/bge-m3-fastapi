from typing import Literal, Optional

from .onnx import *
from .llama_cpp import *

from utils.logger import _logger

logger = _logger(__name__)


def init_runtime(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    batch_size: int = 1,
    max_workers: int = 1,
    backend: Literal["onnx", "llama_cpp"] = "onnx",
    device: Literal["cpu", "cuda"] = "cpu",
):
    match backend:
        case "onnx":
            Embedder = ONNXCpuRuntime if device == "cpu" else ONNXCudaRuntime
            return Embedder(model_path, tokenizer_path, batch_size, max_workers)
        case "llama_cpp":
            if device == "cuda":
                logger(
                    "'llama_cpp' backend supports cpu only.", level="warning"
                )
            return LlamaCppEmbedder(model_path)
