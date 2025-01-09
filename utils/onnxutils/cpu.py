from multiprocessing import cpu_count
from typing import Any, List, Optional
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from schemas.embed import EmbedResult

from .base import ONNXRuntime
from transformers import AutoTokenizer


class ONNXCpuRuntime(ONNXRuntime):

    def _init_tokenizer_session_pool(self):
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = cpu_count() // 2
        sess_options.inter_op_num_threads = cpu_count()

        session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )

        tokenizers = [
            AutoTokenizer.from_pretrained(self.tokenizer_path)
            for _ in range(self.max_workers)
        ]

        return tokenizers, [session]

        )
