from typing import List, overload
import onnxruntime as ort

from schemas.embed import EmbedResult
from .base import ONNXRuntime


class ONNXCudaRuntime(ONNXRuntime):

    def __init__(self, *args):
        super().__init__(*args)
        self.device_type = "cuda"
        self.device_id = 0

        self,
    ):







