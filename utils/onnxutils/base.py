import threading
from typing import List, Literal, overload

from onnxruntime import InferenceSession
import onnxruntime as ort
from transformers import AutoTokenizer

from schemas.embed import EmbedResult


class ONNXRuntime:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if ONNXRuntime._instance is None:
            print("New ONNXRuntime instance has been created")
            ONNXRuntime._instance = super().__new__(cls)
        assert ONNXRuntime._instance is not None
        return ONNXRuntime._instance

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        N: int = 1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.status_event = threading.Event()
        self.status_event.clear()
        self.N = N

    @classmethod
    def get_runtime(cls):
        assert ONNXRuntime._instance is not None
        return ONNXRuntime._instance

    def _inference(
        self,
        session: InferenceSession,
        query: str,
        device_type: Literal["cpu", "cuda"] = "cpu",
    ):
        inputs = self.tokenizer(query, padding="longest", return_tensors="np")
        onnx_inputs = {
            k: ort.OrtValue.ortvalue_from_numpy(v, device_type)
            for k, v in inputs.items()
        }
        outputs = session.run(None, onnx_inputs)
        dense = outputs[0][0].tolist()
        indicies = inputs["input_ids"].tolist()[0]
        weights = outputs[1].squeeze(-1)[0].tolist()
        sparse = {k: w for k, w in zip(indicies, weights)}
        return query, dense, sparse

    @overload
    def parallel_execution(self, queries: str) -> EmbedResult: ...

    @overload
    def parallel_execution(self, queries: List[str]) -> List[EmbedResult]: ...

    def parallel_execution(
        self, queries: str | List[str]
    ) -> EmbedResult | List[EmbedResult]:
        raise NotImplementedError()

    def init_task(self):
        raise NotImplementedError()

    def release_task(self):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()


def init_runtime(
    tokenizer_path: str = "BAAI/bge-m3",
    model_path: str = "models/model.onnx",
    device: Literal["cpu", "cuda"] = "cpu",
    N: int = 2,
) -> ONNXRuntime:
    import os
    from .cpu import ONNXPoolExecutor
    from .cuda import ONNXCudaRuntime

    if device == "cpu" or os.system("nvidia-smi") != 0:
        return ONNXPoolExecutor(tokenizer_path, model_path, N)
    return ONNXCudaRuntime(tokenizer_path, model_path, N)
