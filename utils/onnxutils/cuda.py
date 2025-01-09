from typing import List, overload
import onnxruntime as ort

from schemas.embed import EmbedResult
from .base import ONNXRuntime


# TODO: batch size 늘려서 테스트 필요
class ONNXCudaRuntime(ONNXRuntime):
    """onnx runtime wrapper class for cuda environment"""

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







