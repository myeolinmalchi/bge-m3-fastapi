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

    @overload
    def parallel_execution(self, queries: str) -> EmbedResult: ...

    @overload
    def parallel_execution(self, queries: List[str]) -> List[EmbedResult]: ...

    def parallel_execution(
        self, queries: str | List[str]
    ) -> EmbedResult | List[EmbedResult]:
        _queries = queries if isinstance(queries, list) else [queries]
        results = []
        for query in _queries:
            result = self._inference(self.session, query, device_type="cuda")
            results.append(result)

        return results

    def init_task(self): ...

    def release_task(self): ...

    def release(self):
        del self.session
