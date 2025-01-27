from multiprocessing import cpu_count
from typing import Any, List, Optional
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from transformers import AutoTokenizer
from schemas.embed import EmbedResult

from .base import ONNXEmbedder


class ONNXCudaRuntime(ONNXEmbedder):

    def __init__(self, *args):
        super().__init__(*args)
        self.device_type = "cuda"
        self.device_id = 0

    def _init_session_pool(self):
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = cpu_count() // 2
        providers = [("CUDAExecutionProvider", {"device_id": 0})]

        sessions = [
            ort.InferenceSession(
                self.model_path,
                providers=providers,
                sess_options=sess_options,
            ) for _ in range(self.max_workers)
        ]

        tokenizers = [
            AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
            for _ in range(self.max_workers)
        ]

        return tokenizers, sessions

    def inference(
        self,
        queries: List[str],
        session: Optional[InferenceSession] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ):
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

        io_binding = session.io_binding()
        for k, v in inputs.items():
            ortvalue = ort.OrtValue.ortvalue_from_numpy(v, self.device_type, 0)
            io_binding.bind_input(
                name=k,
                device_type=self.device_type,
                device_id=self.device_id,
                element_type=v.dtype,
                shape=ortvalue.shape(),
                buffer_ptr=ortvalue.data_ptr(),
            )

        output_names = ["dense_vecs", "sparse_vecs", "colbert_vecs"]
        for output_name in output_names:
            io_binding.bind_output(
                name=output_name,
                device_type=self.device_type,
                device_id=self.device_id,
            )

        session.run_with_iobinding(io_binding)
        outputs = io_binding.get_outputs()

        dense_outputs = outputs[0].numpy()
        sparse_outputs = [
            {
                i: w
                for i, w in zip(indicies, weights)
            } for indicies, weights in
            zip(inputs["input_ids"], outputs[1].numpy().squeeze(-1))
        ]

        results = [
            EmbedResult(dense=dense, sparse=sparse, chunk=query) for dense,
            sparse, query in zip(dense_outputs, sparse_outputs, queries)
        ]

        return results
