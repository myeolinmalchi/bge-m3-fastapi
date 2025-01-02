from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue
from typing import List, overload
import onnxruntime as ort

from schemas.embed import EmbedResult
from .base import ONNXRuntime


class ONNXPoolExecutor(ONNXRuntime):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        N: int = 1,
    ):
        super().__init__(tokenizer_path, model_path, N)
        self.sessions = [
            ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            for _ in range(N)
        ]
        self.pool = ThreadPoolExecutor(max_workers=N)
        self.queue = Queue()
        self.num = 0

    @classmethod
    @contextmanager
    def onnx_pool(cls):
        if cls._instance is None:
            raise Exception("ONNXRuntime 인스턴스가 정상적으로 생성되지 않았습니다.")
        cls._instance.init_task()
        yield cls._instance
        cls._instance.release_task()

    @overload
    def parallel_execution(self, queries: str) -> EmbedResult: ...

    @overload
    def parallel_execution(self, queries: List[str]) -> List[EmbedResult]: ...

    def parallel_execution(
        self, queries: str | List[str]
    ) -> EmbedResult | List[EmbedResult]:
        with self.onnx_pool() as pool:
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

    def init_task(self):
        if self.status_event.is_set():
            self.status_event.wait()
        self.status_event.set()

    def release_task(self):
        self.status_event.clear()

    def put(self, query: str):
        future = self.pool.submit(
            self._inference, self.sessions[self.num % self.N], query
        )
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
        for session in self.sessions:
            del session
        del self.tokenizer
