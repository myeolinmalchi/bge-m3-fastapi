from typing import List, overload
import onnxruntime as ort

from schemas.embed import EmbedResult
from .base import ONNXRuntime


class ONNXCudaRuntime(ONNXRuntime):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        N: int = 1,
    ):
        super().__init__(tokenizer_path, model_path, N)
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        self.N = N







