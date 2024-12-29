from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import cpu_count
from queue import Queue
from typing import Callable, List, Optional, overload
import onnxruntime as ort
from onnxruntime import InferenceSession
import threading

from transformers import AutoTokenizer

from schemas.embed import EmbedResult


class ONNXPoolExecutor:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model_path: str = "./models/model.onnx",
        N: int = 1,
        func: Optional[Callable] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sessions = self._init_sessions(model_path, N)
        self.session = ort.InferenceSession(model_path)
        self.pool = ThreadPoolExecutor(max_workers=N)
        self.func = func or self._default_inference
        self.status_event = threading.Event()
        self.status_event.clear()
        self.queue = Queue()
        self.num = 0
        self.N = N

    def _init_sessions(self, model_path: str, N: int):
        return [ort.InferenceSession(model_path) for _ in range(N)]

    def _default_inference(self, session: InferenceSession, query: str):
        inputs = self.tokenizer(query, padding="longest", return_tensors="np")
        onnx_inputs = {
            k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()
        }
        outputs = session.run(None, onnx_inputs)
        dense = outputs[0][0].tolist()
        indicies = inputs["input_ids"].tolist()[0]
        weights = outputs[1].squeeze(-1)[0].tolist()
        sparse = {k: w for k, w in zip(indicies, weights)}
        return query, dense, sparse

    def init_task(self):
        if self.status_event.is_set():
            self.status_event.wait()
        self.status_event.set()

    def release_task(self):
        self.status_event.clear()

    def put(self, query: str):
        future = self.pool.submit(self.func, self.sessions[self.num % self.N], query)
        self.queue.put(future)
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None
        future = self.queue.get()
        result = future.result()
        return result

    def release(self):
        self.pool.shutdown()
        del self.session
        del self.tokenizer


_pool: Optional[ONNXPoolExecutor] = None


@contextmanager
def onnx_pool(
    model_name: str = "BAAI/bge-m3",
    model_path: str = "models/model.onnx",
    N: Optional[int] = None,
):
    global _pool
    if _pool == None:
        # worker_n = cpu_count() if N is None else N
        worker_n = 2 if N is None else N
        _pool = ONNXPoolExecutor(model_name, model_path, worker_n)
    _pool.init_task()
    yield _pool
    _pool.release_task()


@overload
def run_onnx_pool(queries: List[str]) -> List[EmbedResult]: ...


@overload
def run_onnx_pool(queries: str) -> EmbedResult: ...


def run_onnx_pool(queries: List[str] | str):
    with onnx_pool() as pool:
        if isinstance(queries, str):
            query = queries
            pool.put(query)
            result = pool.get()
            if result is None:
                raise Exception("Error has been occured (ONNX)")
            _, dense, sparse = result
            return EmbedResult(dense=dense, sparse=sparse)

        results: List[EmbedResult] = []
        num = len(queries)
        for idx in range(num + min(pool.N, num)):
            if idx >= min(pool.N, num):
                result = pool.get()
                if result is None:
                    raise Exception("Error has been occured (ONNX)")
                query, dense, sparse = result
                results.append(EmbedResult(dense=dense, sparse=sparse, chunk=query))
            if idx < num:
                query = queries[idx]
                pool.put(query)
        return results
