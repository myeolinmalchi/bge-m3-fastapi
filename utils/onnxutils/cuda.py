from typing import List, overload
import onnxruntime as ort

from schemas.embed import EmbedResult
from .base import ONNXRuntime


class ONNXCudaRuntime(ONNXRuntime):

    def __init__(self, *args):
        super().__init__(*args)
        self.device_type = "cuda"
        self.device_id = 0

    def _init_tokenizer_session_pool(self):
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = cpu_count() // 2
        providers = [("CUDAExecutionProvider", {"device_id": 0})]

        sessions = [
            ort.InferenceSession(
                self.model_path,
                providers=providers,
                sess_options=sess_options,
            )
            for _ in range(self.max_workers)
        ]

        tokenizers = [
            AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
            for _ in range(self.max_workers)
        ]

        return tokenizers, sessions

        self,
    ):







