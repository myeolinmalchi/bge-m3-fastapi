from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Literal, Optional, Tuple

from onnxruntime import InferenceSession
import asyncio

from schemas.embed import EmbedResult


class ONNXRuntime:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls == ONNXRuntime:
            raise Exception("ONNXRuntime 인스턴스는 직접 생성할 수 없습니다.")

        if ONNXRuntime._instance is None:
            print("New ONNXRuntime instance has been created")
            ONNXRuntime._instance = super().__new__(cls)

        return ONNXRuntime._instance

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        batch_size: int = 1,
        max_workers: int = 1,
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        temp = self._init_tokenizer_session_pool()
        self._tokenizers = temp[0]
        self._sessions = temp[1]

        self._session_index = 0
        self._tokenizer_index = 0

    @abstractmethod
    def _init_tokenizer_session_pool(self) -> Tuple[List, List[InferenceSession]]:
        raise NotImplementedError(
            "method '_init_tokenizer_session_pool' must be implemented"
        )

    @property
    def session(self) -> InferenceSession:
        session = self._sessions[self._session_index % self.max_workers]
        self._session_index += 1
        return session

    @property
    def tokenizer(self):
        tokenizer = self._tokenizers[self._tokenizer_index % self.max_workers]
        self._tokenizer_index += 1
        return tokenizer

    @classmethod
    def get_runtime(cls):
        if ONNXRuntime._instance is None:
            raise Exception("'init_runtime'을 먼저 호출하세요.")
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
