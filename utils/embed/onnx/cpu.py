from multiprocessing import cpu_count
from typing import Any, List, Optional
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from schemas.embed import EmbedResult

from .base import ONNXEmbedder
from transformers import AutoTokenizer


class ONNXCpuRuntime(ONNXEmbedder):

    def _init_session_pool(self):
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = cpu_count() // 2
        sess_options.inter_op_num_threads = cpu_count()

        sessions = [
            ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            ) for _ in range(self.max_workers)
        ]

        tokenizers = [
            AutoTokenizer.from_pretrained(self.tokenizer_path)
            for _ in range(self.max_workers)
        ]

        return sessions, tokenizers

    def inference(
        self,
        queries: List[str],
        session: Optional[InferenceSession] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> List[EmbedResult]:
        if len(queries) > self.batch_size:
            raise Exception(
                f"최대 배치 크기를 초과한 입력({len(queries)} > {self.max_workers})입니다."
            )

        _session, _tokenizer = self.session
        session = _session if session is None else session
        tokenizer = _tokenizer if tokenizer is None else tokenizer

        if not tokenizer:
            raise ValueError("Tokenizer is not initialized.")

        inputs = tokenizer(
            queries,
            padding="longest",
            return_tensors="np",
            truncation=True,
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        outputs = session.run(None, onnx_inputs)

        dense_outputs = outputs[0]
        sparse_outputs = [{
            i: w
            for i, w in zip(indicies, weights)
        } for indicies, weights in
                          zip(inputs["input_ids"], outputs[1].squeeze(-1))]

        results = [
            EmbedResult(dense=dense, sparse=sparse, chunk=query) for dense,
            sparse, query in zip(dense_outputs, sparse_outputs, queries)
        ]

        return results
