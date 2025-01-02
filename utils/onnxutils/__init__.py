from .base import *
from .cpu import *
from .cuda import *

from typing import Literal
import os


def init_runtime(
    tokenizer_path: str = "BAAI/bge-m3",
    model_path: str = "models/model.onnx",
    device: Literal["cpu", "cuda"] = "cpu",
    N: int = 2,
):
    if device == "cpu" or os.system("nvidia-smi") != 0:
        return ONNXPoolExecutor(tokenizer_path, model_path, N)
    return ONNXCudaRuntime(tokenizer_path, model_path, N)
